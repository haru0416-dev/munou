//! Conversation corpus: a single `u32` stream + suffix array + generation buffer.
//!
//! Variable-length n-grams are looked up by binary search on the SA (body)
//! and by a linear scan of the small generation buffer. When the buffer
//! exceeds `merge_threshold`, the body is rebuilt with SA-IS.

use rustc_hash::FxHashMap;

use crate::alias::AliasTable;
use crate::ids::{TokenId, EOS, SEP};
use crate::sais::{sa_range, suffix_array};

#[derive(Debug, Clone, Default)]
pub struct Store {
    /// Committed token stream (Markov alphabet = chunk ids + specials).
    pub text: Vec<TokenId>,
    pub sa: Vec<u32>,
    /// Unmerged recent tokens.
    pub buf: Vec<TokenId>,
    /// Unigram counts over body+buf (includes specials).
    unigram: FxHashMap<TokenId, u32>,
    /// Continuation counts: number of unique left contexts per token (KN unigram).
    continuation: FxHashMap<TokenId, u32>,
    /// Bigram counts over body+buf, kept incrementally so `continuation` and
    /// the count-of-counts below include buffered utterances (they used to be
    /// stale until the next merge).
    bigrams: FxHashMap<(TokenId, TokenId), u32>,
    /// Bigram count-of-counts n1..n4 for Chen-Goodman modified KN.
    bigram_n1: u64,
    bigram_n2: u64,
    bigram_n3: u64,
    bigram_n4: u64,
    merge_threshold: usize,
    /// Lazily rebuilt copies of `ml_unigram` / `continuation_unigram`.
    /// Invalidated on any corpus mutation; identical values to the uncached fns.
    ml_cache: Option<Vec<(TokenId, f64)>>,
    cont_cache: Option<Vec<(TokenId, f64)>>,
    /// id→prob map and alias table over the cached unigram, for O(1) tail
    /// sampling in the sparse generation path.
    ml_aux: Option<(FxHashMap<TokenId, f64>, AliasTable)>,
    cont_aux: Option<(FxHashMap<TokenId, f64>, AliasTable)>,
}

/// Borrowed view of the cached sampling unigram: the id-sorted distribution,
/// an id→prob map, and an alias table for O(1) draws from the full unigram.
#[derive(Clone, Copy)]
pub struct SamplingUnigram<'a> {
    pub dist: &'a [(TokenId, f64)],
    pub map: &'a FxHashMap<TokenId, f64>,
    pub alias: &'a AliasTable,
}

impl Store {
    pub fn new(merge_threshold: usize) -> Self {
        Self {
            merge_threshold: merge_threshold.max(32),
            ..Self::default()
        }
    }

    pub fn len(&self) -> usize {
        self.text.len() + self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty() && self.buf.is_empty()
    }

    /// Append one utterance (already interned chunks), wrapping with SEP..EOS.
    pub fn push_utterance(&mut self, chunks: &[TokenId]) {
        if chunks.is_empty() {
            return;
        }
        let prev_last = self
            .buf
            .last()
            .copied()
            .or_else(|| self.text.last().copied());
        self.buf.push(SEP);
        for &t in chunks {
            self.buf.push(t);
        }
        self.buf.push(EOS);
        if self.buf.len() >= self.merge_threshold {
            self.merge();
        } else {
            let start = self.buf.len().saturating_sub(chunks.len() + 2);
            let slice = self.buf[start..].to_vec();
            self.bump_unigram_slice(&slice);
            self.bump_bigram_slice(prev_last, &slice);
            self.invalidate_unigram_caches();
        }
    }

    /// Append without stats or merging. For log replay only: no lookups happen
    /// until the single final `merge`, whose recompute makes the end state
    /// identical to per-utterance pushes — without rebuilding the SA every
    /// `merge_threshold` tokens (that made `Engine::open` quadratic in N).
    pub(crate) fn push_utterance_deferred(&mut self, chunks: &[TokenId]) {
        if chunks.is_empty() {
            return;
        }
        self.buf.push(SEP);
        self.buf.extend_from_slice(chunks);
        self.buf.push(EOS);
        self.invalidate_unigram_caches();
    }

    fn bump_unigram_slice(&mut self, slice: &[TokenId]) {
        for &t in slice {
            *self.unigram.entry(t).or_insert(0) += 1;
        }
    }

    fn bump_bigram_slice(&mut self, prev_last: Option<TokenId>, slice: &[TokenId]) {
        if let (Some(l), Some(&w)) = (prev_last, slice.first()) {
            self.bump_bigram(l, w);
        }
        for pair in slice.windows(2) {
            self.bump_bigram(pair[0], pair[1]);
        }
    }

    /// One bigram observation: move its count-of-counts bin and, on first
    /// sight, the continuation count. Matches `recompute_stats` bin for bin.
    fn bump_bigram(&mut self, l: TokenId, w: TokenId) {
        let c = self.bigrams.entry((l, w)).or_insert(0);
        *c += 1;
        match *c {
            1 => {
                *self.continuation.entry(w).or_insert(0) += 1;
                self.bigram_n1 += 1;
            }
            2 => {
                self.bigram_n1 -= 1;
                self.bigram_n2 += 1;
            }
            3 => {
                self.bigram_n2 -= 1;
                self.bigram_n3 += 1;
            }
            4 => {
                self.bigram_n3 -= 1;
                self.bigram_n4 += 1;
            }
            5 => {
                self.bigram_n4 -= 1;
            }
            _ => {}
        }
    }

    pub fn merge(&mut self) {
        if self.buf.is_empty() && !self.text.is_empty() && !self.sa.is_empty() {
            return;
        }
        self.text.extend_from_slice(&self.buf);
        self.buf.clear();
        // Replay can leave a corpus-sized capacity behind; keep one
        // generation's worth.
        self.buf.shrink_to(self.merge_threshold);
        self.rebuild_index();
    }

    pub fn rebuild_index(&mut self) {
        self.sa = suffix_array(&self.text);
        self.recompute_stats();
    }

    fn recompute_stats(&mut self) {
        self.invalidate_unigram_caches();
        // One pass over text⧺buf with a `prev` cursor — no stream copy.
        let mut unigram = std::mem::take(&mut self.unigram);
        let mut bigrams = std::mem::take(&mut self.bigrams);
        unigram.clear();
        bigrams.clear();
        let mut prev: Option<TokenId> = None;
        for &t in self.text.iter().chain(self.buf.iter()) {
            *unigram.entry(t).or_insert(0) += 1;
            if let Some(l) = prev {
                *bigrams.entry((l, t)).or_insert(0) += 1;
            }
            prev = Some(t);
        }
        self.unigram = unigram;
        // continuation(w) = unique left contexts = distinct (·,w) bigram keys.
        self.continuation.clear();
        self.bigram_n1 = 0;
        self.bigram_n2 = 0;
        self.bigram_n3 = 0;
        self.bigram_n4 = 0;
        for (&(_, w), &c) in &bigrams {
            *self.continuation.entry(w).or_insert(0) += 1;
            match c {
                1 => self.bigram_n1 += 1,
                2 => self.bigram_n2 += 1,
                3 => self.bigram_n3 += 1,
                4 => self.bigram_n4 += 1,
                _ => {}
            }
        }
        self.bigrams = bigrams;
    }

    /// Chen-Goodman D1/D2/D3+ from this corpus. Not an ARPA / KenLM file.
    pub fn mkn_discounts(&self, fallback: f64) -> (f64, f64, f64) {
        crate::smoothing::chen_goodman(
            self.bigram_n1,
            self.bigram_n2,
            self.bigram_n3,
            self.bigram_n4,
            fallback,
        )
    }

    pub fn bigram_count_of_counts(&self) -> (u64, u64, u64, u64) {
        (
            self.bigram_n1,
            self.bigram_n2,
            self.bigram_n3,
            self.bigram_n4,
        )
    }

    /// Next-token counts for an exact context, combining SA body and buffer.
    ///
    /// Inside the SA range all suffixes share `ctx`, so they are ordered by the
    /// token at offset `ctx.len()`; distinct next tokens form contiguous runs
    /// whose ends are found by binary search — O(distinct·log occ), not O(occ).
    pub fn next_counts(&self, ctx: &[TokenId]) -> (Vec<(TokenId, u32)>, u32) {
        if ctx.is_empty() {
            return (Vec::new(), 0);
        }
        let mut body: Vec<(TokenId, u32)> = Vec::new();
        if let Some((lo, hi)) = sa_range(&self.text, &self.sa, ctx) {
            let off = ctx.len();
            let n = self.text.len();
            if hi - lo <= 64 {
                // Small ranges: one linear pass beats per-run binary search.
                // Next tokens appear as ascending contiguous runs, so run-length
                // counting yields the same id-sorted list as the branch below.
                let mut cur: Option<(TokenId, u32)> = None;
                for &p in &self.sa[lo..hi] {
                    let j = p as usize + off;
                    if j >= n {
                        continue;
                    }
                    let t = self.text[j];
                    cur = match cur {
                        Some((tc, c)) if tc == t => Some((tc, c + 1)),
                        Some(done) => {
                            body.push(done);
                            Some((t, 1))
                        }
                        None => Some((t, 1)),
                    };
                }
                if let Some(done) = cur {
                    body.push(done);
                }
                return self.finish_counts(body, ctx);
            }
            let mut r = lo;
            // The suffix equal to `ctx` at end-of-text sorts first; it has no next token.
            while r < hi && self.sa[r] as usize + off >= n {
                r += 1;
            }
            while r < hi {
                let t = self.text[self.sa[r] as usize + off];
                let mut a = r + 1;
                let mut b = hi;
                while a < b {
                    let m = (a + b) / 2;
                    let p = self.sa[m] as usize + off;
                    if p < n && self.text[p] == t {
                        a = m + 1;
                    } else {
                        b = m;
                    }
                }
                body.push((t, (a - r) as u32));
                r = a;
            }
        }
        self.finish_counts(body, ctx)
    }

    /// Fold the generation buffer into the SA-body counts.
    fn finish_counts(
        &self,
        body: Vec<(TokenId, u32)>,
        ctx: &[TokenId],
    ) -> (Vec<(TokenId, u32)>, u32) {
        let mut acc: FxHashMap<TokenId, u32> = FxHashMap::default();
        count_linear(&self.buf, ctx, &mut acc);
        if acc.is_empty() {
            let total: u32 = body.iter().map(|(_, c)| *c).sum();
            return (body, total);
        }
        for (t, c) in body {
            *acc.entry(t).or_insert(0) += c;
        }
        let total: u32 = acc.values().copied().sum();
        let mut v: Vec<(TokenId, u32)> = acc.into_iter().collect();
        v.sort_by_key(|(id, _)| *id);
        (v, total)
    }

    fn invalidate_unigram_caches(&mut self) {
        self.ml_cache = None;
        self.cont_cache = None;
        self.ml_aux = None;
        self.cont_aux = None;
    }

    /// Cached copy of `continuation_unigram` (kn) or `ml_unigram` (naive).
    /// Same values; rebuilt only after the corpus changed.
    pub fn sampling_unigram(&mut self, kn: bool) -> Vec<(TokenId, f64)> {
        self.warm_sampling(kn);
        if kn {
            self.cont_cache.clone().unwrap_or_default()
        } else {
            self.ml_cache.clone().unwrap_or_default()
        }
    }

    /// Build the unigram caches (distribution + map + alias) if stale.
    pub fn warm_sampling(&mut self, kn: bool) {
        if kn {
            if self.cont_cache.is_none() {
                self.cont_cache = Some(self.continuation_unigram());
            }
            if self.cont_aux.is_none() {
                let d = self.cont_cache.as_deref().unwrap_or(&[]);
                self.cont_aux = Some(build_aux(d));
            }
        } else {
            if self.ml_cache.is_none() {
                self.ml_cache = Some(self.ml_unigram());
            }
            if self.ml_aux.is_none() {
                let d = self.ml_cache.as_deref().unwrap_or(&[]);
                self.ml_aux = Some(build_aux(d));
            }
        }
    }

    /// Borrowed view of the warmed caches. Call `warm_sampling` first;
    /// returns `None` when the caches are stale.
    pub fn sampling_view(&self, kn: bool) -> Option<SamplingUnigram<'_>> {
        let (dist, aux) = if kn {
            (self.cont_cache.as_deref()?, self.cont_aux.as_ref()?)
        } else {
            (self.ml_cache.as_deref()?, self.ml_aux.as_ref()?)
        };
        Some(SamplingUnigram {
            dist,
            map: &aux.0,
            alias: &aux.1,
        })
    }

    /// Whether the exact token run occurs anywhere in body or buffer.
    pub fn contains_seq(&self, pat: &[TokenId]) -> bool {
        !pat.is_empty() && occurs(self, pat)
    }

    /// Corpus count of a single token (0 if unseen).
    pub fn count_of(&self, id: TokenId) -> u32 {
        self.unigram.get(&id).copied().unwrap_or(0)
    }

    pub fn unigram_counts(&self) -> Vec<(TokenId, u32)> {
        let mut v: Vec<(TokenId, u32)> = self.unigram.iter().map(|(&k, &c)| (k, c)).collect();
        v.sort_by_key(|(id, _)| *id);
        v
    }

    /// Kneser-Ney unigram: P_cont(w) ∝ unique left contexts of w.
    pub fn continuation_unigram(&self) -> Vec<(TokenId, f64)> {
        let total: u32 = self.continuation.values().copied().sum();
        if total == 0 {
            return self.ml_unigram();
        }
        let t = total as f64;
        let mut v: Vec<(TokenId, f64)> = self
            .continuation
            .iter()
            .map(|(&id, &c)| (id, c as f64 / t))
            .collect();
        v.sort_by_key(|(id, _)| *id);
        v
    }

    pub fn ml_unigram(&self) -> Vec<(TokenId, f64)> {
        let total: u32 = self.unigram.values().copied().sum();
        if total == 0 {
            return Vec::new();
        }
        let t = total as f64;
        self.unigram_counts()
            .into_iter()
            .map(|(id, c)| (id, c as f64 / t))
            .collect()
    }

    pub fn longest_match_len(&self, hay: &[TokenId]) -> usize {
        if hay.is_empty() {
            return 0;
        }
        // longest prefix of `hay` that occurs in the corpus
        let mut best = 0usize;
        for n in (1..=hay.len()).rev() {
            let pat = &hay[..n];
            let (c, total) = self.next_counts(pat);
            let _ = c;
            if total > 0 || occurs(self, pat) {
                best = n;
                break;
            }
        }
        best
    }
}

fn build_aux(dist: &[(TokenId, f64)]) -> (FxHashMap<TokenId, f64>, AliasTable) {
    let map: FxHashMap<TokenId, f64> = dist.iter().copied().collect();
    let weights: Vec<f64> = dist.iter().map(|(_, p)| *p).collect();
    (map, AliasTable::from_weights(&weights))
}

fn occurs(store: &Store, pat: &[TokenId]) -> bool {
    if sa_range(&store.text, &store.sa, pat).is_some() {
        return true;
    }
    store.buf.windows(pat.len()).any(|w| w == pat)
}

fn count_linear(text: &[TokenId], ctx: &[TokenId], acc: &mut FxHashMap<TokenId, u32>) {
    if ctx.len() >= text.len() {
        return;
    }
    for i in 0..=text.len() - ctx.len() {
        if &text[i..i + ctx.len()] == ctx {
            let j = i + ctx.len();
            if j < text.len() {
                *acc.entry(text[j]).or_insert(0) += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_after_merge() {
        let mut s = Store::new(4);
        s.push_utterance(&[10, 11, 12]);
        s.push_utterance(&[10, 11, 13]);
        s.merge();
        let (c, total) = s.next_counts(&[10, 11]);
        assert!(total >= 2);
        let ids: Vec<TokenId> = c.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&12) || ids.contains(&13));
    }

    /// Incremental (buffered) stats must equal a from-scratch recompute over
    /// the same stream — KN continuation and count-of-counts are no longer
    /// stale between merges.
    #[test]
    fn incremental_stats_match_recompute() {
        let mut a = Store::new(1024); // stays buffered
        let mut b = Store::new(1024);
        let utts: Vec<Vec<TokenId>> = vec![
            vec![20, 21, 22],
            vec![20, 21],
            vec![23, 20, 21, 22],
            vec![24],
            vec![20, 21, 22],
        ];
        for u in &utts {
            a.push_utterance(u);
            b.push_utterance(u);
        }
        b.merge();
        assert_eq!(a.bigram_count_of_counts(), b.bigram_count_of_counts());
        assert_eq!(a.continuation_unigram(), b.continuation_unigram());
        assert_eq!(a.unigram_counts(), b.unigram_counts());
    }

    /// Deferred replay pushes + one final merge must be bit-identical to
    /// per-utterance pushes with periodic merges.
    #[test]
    fn deferred_replay_equals_incremental_pushes() {
        let mut a = Store::new(8); // tiny threshold → many merges on the way
        let mut b = Store::new(8);
        let mut utts: Vec<Vec<TokenId>> = Vec::new();
        for i in 0..50u32 {
            utts.push(vec![20 + i % 7, 30 + i % 5, 40 + i % 3]);
        }
        for u in &utts {
            a.push_utterance(u);
        }
        a.merge();
        for u in &utts {
            b.push_utterance_deferred(u);
        }
        b.merge();
        assert_eq!(a.text, b.text);
        assert_eq!(a.sa, b.sa);
        assert_eq!(a.unigram_counts(), b.unigram_counts());
        assert_eq!(a.continuation_unigram(), b.continuation_unigram());
        assert_eq!(a.bigram_count_of_counts(), b.bigram_count_of_counts());
    }

    fn naive_counts(s: &Store, ctx: &[TokenId]) -> (Vec<(TokenId, u32)>, u32) {
        let mut acc: FxHashMap<TokenId, u32> = FxHashMap::default();
        count_linear(&s.text, ctx, &mut acc);
        count_linear(&s.buf, ctx, &mut acc);
        let total: u32 = acc.values().copied().sum();
        let mut v: Vec<(TokenId, u32)> = acc.into_iter().collect();
        v.sort_by_key(|(id, _)| *id);
        (v, total)
    }

    /// The small-range linear path and the run-binary-search path must both
    /// match a naive window scan.
    #[test]
    fn next_counts_linear_and_binary_paths_agree() {
        let mut s = Store::new(32);
        for i in 0..100u32 {
            s.push_utterance(&[50, 51, 52 + (i % 3)]);
        }
        s.push_utterance(&[60, 61, 62]);
        s.merge();
        for ctx in [vec![50u32, 51], vec![60, 61], vec![51]] {
            let (got, total) = s.next_counts(&ctx);
            let (want, wtotal) = naive_counts(&s, &ctx);
            assert_eq!(got, want, "ctx={ctx:?}");
            assert_eq!(total, wtotal, "ctx={ctx:?}");
        }
    }

    #[test]
    fn mkn_discounts_increase_with_count_bin() {
        let mut s = Store::new(32);
        for i in 0..40u32 {
            s.push_utterance(&[100, 200 + i]);
        }
        for _ in 0..8 {
            s.push_utterance(&[10, 11]);
        }
        for _ in 0..3 {
            s.push_utterance(&[12, 13]);
        }
        for _ in 0..2 {
            s.push_utterance(&[14, 15]);
        }
        s.merge();
        let (n1, n2, n3, n4) = s.bigram_count_of_counts();
        assert!(n1 > 0 && n2 > 0, "n1={n1} n2={n2} n3={n3} n4={n4}");
        let (d1, d2, d3) = s.mkn_discounts(0.75);
        assert!((0.05..=0.95).contains(&d1), "d1={d1}");
        assert!((0.05..=1.95).contains(&d2), "d2={d2}");
        assert!((0.05..=2.95).contains(&d3), "d3={d3}");
    }
}
