//! Conversation corpus as a **weighted multiset of distinct utterances**.
//!
//! The closed loop repeats the same lines heavily (fabricate unique0: 46
//! distinct in 200k utterances; even unique0.3 logs are 72% duplicates), and
//! every pattern the engine looks up is special-free — generation contexts
//! come from chunk history, so no query can cross an utterance boundary
//! (crossing would require SEP/EOS inside the pattern). Under those two
//! invariants every statistic decomposes over distinct utterances:
//!
//! - counts of a pattern = Σ occurrences inside distinct utterance × its
//!   multiplicity,
//! - the bigram stream = Σ per-utterance internal pairs × multiplicity, plus
//!   the boundary pair (EOS, SEP) exactly (total utterances − 1) times,
//!
//! so the suffix array only ever indexes the *deduplicated* stream. Repeats
//! bump a counter instead of growing the SA: on repetitive logs rebuilds all
//! but disappear and the index stays a few hundred tokens. All returned
//! values are identical to a flat-stream store (the reference test below
//! pins them against a flat replay).
//!
//! Precondition (debug-asserted): lookup patterns never contain SEP/EOS.
//! Order-dependent cross-utterance patterns are unrepresentable here; the
//! engine never issues them.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::alias::AliasTable;
use crate::ids::{TokenId, EOS, SEP};
use crate::sais::{sa_range, suffix_array};

#[derive(Debug, Clone, Default)]
pub struct Store {
    /// Committed dedup stream: each distinct utterance once, wrapped SEP..EOS.
    pub text: Vec<TokenId>,
    pub sa: Vec<u32>,
    /// Pending dedup stream: new distinct utterances awaiting a merge.
    pub buf: Vec<TokenId>,
    /// Distinct utterances (unwrapped chunks) with multiplicities, in first-
    /// appearance order. The Arc is shared with the index key.
    utts: Vec<(Arc<[TokenId]>, u32)>,
    utt_index: FxHashMap<Arc<[TokenId]>, usize>,
    /// Start offsets in `text` of the first `committed` distinct utterances.
    offsets: Vec<u32>,
    /// Utterance index per committed text position (position → distinct utt).
    pos_utt: Vec<u32>,
    /// Multiplicity prefix sums in SA order: wprefix[i] = Σ weight(sa[j]) for
    /// j < i, where weight = multiplicity of the utterance owning sa[j].
    /// Lets a next-token run [r, a) sum its weighted count as a difference —
    /// rebuilt lazily because multiplicities change without touching the SA.
    wprefix: Vec<u64>,
    wprefix_stale: bool,
    /// Start offsets in `buf` of the pending distinct utterances.
    pending_offsets: Vec<u32>,
    committed: usize,
    /// Weighted totals (multiplicities included).
    total_tokens: usize,
    total_utts: u64,
    /// Deferred pushes skip stat bumps; merge recomputes when set.
    stats_dirty: bool,
    /// Weighted unigram counts over the whole multiset (includes specials).
    unigram: FxHashMap<TokenId, u32>,
    /// Continuation counts: number of unique left contexts per token (KN unigram).
    continuation: FxHashMap<TokenId, u32>,
    /// Weighted bigram counts (kept incrementally; includes the (EOS,SEP)
    /// boundary pair).
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

    /// Weighted token count (what the gauges call `tokens`).
    pub fn len(&self) -> usize {
        self.total_tokens
    }

    pub fn is_empty(&self) -> bool {
        self.total_utts == 0
    }

    /// Append one utterance (already interned chunks). A repeat of a known
    /// utterance only bumps its multiplicity — the SA never sees it.
    pub fn push_utterance(&mut self, chunks: &[TokenId]) {
        if chunks.is_empty() {
            return;
        }
        let had_prev = self.total_utts > 0;
        self.register(chunks);
        self.bump_stats(chunks, had_prev);
        self.invalidate_unigram_caches();
        if self.buf.len() >= self.merge_threshold {
            self.merge();
        } else if self.wprefix_stale {
            self.refresh_wprefix();
        }
    }

    /// Append without stat bumps or merging. For log replay only: no lookups
    /// happen until the single final `merge`, whose recompute makes the end
    /// state identical to per-utterance pushes.
    pub(crate) fn push_utterance_deferred(&mut self, chunks: &[TokenId]) {
        if chunks.is_empty() {
            return;
        }
        self.register(chunks);
        self.stats_dirty = true;
        self.invalidate_unigram_caches();
    }

    /// Dedup bookkeeping shared by both push paths.
    fn register(&mut self, chunks: &[TokenId]) {
        if let Some(&i) = self.utt_index.get(chunks) {
            self.utts[i].1 += 1;
            if i < self.committed {
                // A committed utterance's multiplicity changed; the SA-order
                // weight prefix sums no longer match.
                self.wprefix_stale = true;
            }
        } else {
            let key: Arc<[TokenId]> = Arc::from(chunks);
            let i = self.utts.len();
            self.utts.push((key.clone(), 1));
            self.utt_index.insert(key, i);
            self.pending_offsets.push(self.buf.len() as u32);
            self.buf.push(SEP);
            self.buf.extend_from_slice(chunks);
            self.buf.push(EOS);
        }
        self.total_utts += 1;
        self.total_tokens += chunks.len() + 2;
    }

    /// One more weighted occurrence of `chunks`: unigram, internal bigrams,
    /// and the (EOS, SEP) boundary against the previous utterance.
    fn bump_stats(&mut self, chunks: &[TokenId], had_prev: bool) {
        *self.unigram.entry(SEP).or_insert(0) += 1;
        for &t in chunks {
            *self.unigram.entry(t).or_insert(0) += 1;
        }
        *self.unigram.entry(EOS).or_insert(0) += 1;
        if had_prev {
            self.bump_bigram(EOS, SEP);
        }
        self.bump_bigram(SEP, chunks[0]);
        for pair in chunks.windows(2) {
            self.bump_bigram(pair[0], pair[1]);
        }
        self.bump_bigram(chunks[chunks.len() - 1], EOS);
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
        if self.buf.is_empty() && !self.stats_dirty {
            if !self.text.is_empty() && !self.sa.is_empty() {
                return;
            }
            if self.text.is_empty() {
                // Nothing committed and nothing pending.
                self.recompute_stats();
                return;
            }
        }
        if !self.buf.is_empty() {
            let base = self.text.len() as u32;
            for &off in &self.pending_offsets {
                self.offsets.push(base + off);
            }
            self.pending_offsets.clear();
            self.text.extend_from_slice(&self.buf);
            self.buf.clear();
            self.buf.shrink_to(self.merge_threshold);
            self.committed = self.utts.len();
        }
        self.rebuild_index();
    }

    pub fn rebuild_index(&mut self) {
        self.sa = suffix_array(&self.text);
        // position → owning distinct utterance, for occurrence weighting
        self.pos_utt = vec![0; self.text.len()];
        for (i, &off) in self.offsets.iter().enumerate() {
            let end = self
                .offsets
                .get(i + 1)
                .map(|&o| o as usize)
                .unwrap_or(self.text.len());
            for p in off as usize..end {
                self.pos_utt[p] = i as u32;
            }
        }
        self.refresh_wprefix();
        self.recompute_stats();
        self.stats_dirty = false;
    }

    /// Rebuild the SA-order multiplicity prefix sums. O(committed tokens);
    /// runs on merge and whenever a committed utterance's count changed.
    fn refresh_wprefix(&mut self) {
        self.wprefix.clear();
        self.wprefix.reserve(self.sa.len() + 1);
        let mut acc = 0u64;
        self.wprefix.push(0);
        for &p in &self.sa {
            acc += self.utts[self.pos_utt[p as usize] as usize].1 as u64;
            self.wprefix.push(acc);
        }
        self.wprefix_stale = false;
    }

    /// Weighted recompute from the distinct-utterance multiset. Identical to
    /// a flat-stream pass: internal pairs scale with multiplicity and the
    /// boundary pair (EOS, SEP) appears exactly total_utts − 1 times,
    /// independent of arrival order.
    fn recompute_stats(&mut self) {
        self.invalidate_unigram_caches();
        let mut unigram = std::mem::take(&mut self.unigram);
        let mut bigrams = std::mem::take(&mut self.bigrams);
        unigram.clear();
        bigrams.clear();
        for (chunks, k) in &self.utts {
            let k = *k;
            *unigram.entry(SEP).or_insert(0) += k;
            *unigram.entry(EOS).or_insert(0) += k;
            for &t in chunks.iter() {
                *unigram.entry(t).or_insert(0) += k;
            }
            *bigrams.entry((SEP, chunks[0])).or_insert(0) += k;
            for pair in chunks.windows(2) {
                *bigrams.entry((pair[0], pair[1])).or_insert(0) += k;
            }
            *bigrams.entry((chunks[chunks.len() - 1], EOS)).or_insert(0) += k;
        }
        if self.total_utts >= 2 {
            *bigrams.entry((EOS, SEP)).or_insert(0) += (self.total_utts - 1) as u32;
        }
        self.unigram = unigram;
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

    /// Chen-Goodman D1/D2/D3+ from this corpus.
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

    /// Multiplicity of the distinct utterance a position belongs to.
    fn weight_at(offsets: &[u32], utts: &[(Arc<[TokenId]>, u32)], first: usize, pos: usize) -> u32 {
        let i = offsets.partition_point(|&o| o as usize <= pos);
        utts[first + i - 1].1
    }

    /// Next-token counts for an exact context: occurrences in the dedup
    /// stream, each weighted by its utterance's multiplicity. Patterns are
    /// special-free (see module docs), so every occurrence lies inside one
    /// distinct utterance and the attribution is exact.
    pub fn next_counts(&self, ctx: &[TokenId]) -> (Vec<(TokenId, u32)>, u32) {
        if ctx.is_empty() {
            return (Vec::new(), 0);
        }
        // Precondition: no EOS in lookup patterns (the engine never emits it
        // into a context). SEP is fine: in the stream SEP is always preceded
        // by EOS, so a chunk directly followed by SEP matches nowhere in
        // either representation, and a pattern-leading SEP matches only at
        // block starts — entirely inside one distinct utterance. Under that,
        // dedup counting is exact.
        debug_assert!(
            !ctx.contains(&EOS),
            "dedup store precondition: lookup patterns never contain EOS"
        );
        debug_assert!(
            !self.wprefix_stale,
            "wprefix refreshed on every mutation entry point"
        );
        let mut body: Vec<(TokenId, u32)> = Vec::new();
        if let Some((lo, hi)) = sa_range(&self.text, &self.sa, ctx) {
            let off = ctx.len();
            let n = self.text.len();
            let mut r = lo;
            // The suffix equal to `ctx` at end-of-text sorts first; no next token.
            while r < hi && self.sa[r] as usize + off >= n {
                r += 1;
            }
            if hi - r <= 64 {
                // Small ranges: linear pass with per-position weights. Runs
                // are ascending by next token, so run-length accumulation
                // yields the same id-sorted list as the branch below.
                let mut cur: Option<(TokenId, u32)> = None;
                for &p in &self.sa[r..hi] {
                    let p = p as usize;
                    let t = self.text[p + off];
                    let w = self.utts[self.pos_utt[p] as usize].1;
                    cur = match cur {
                        Some((tc, c)) if tc == t => Some((tc, c + w)),
                        Some(done) => {
                            body.push(done);
                            Some((t, w))
                        }
                        None => Some((t, w)),
                    };
                }
                if let Some(done) = cur {
                    body.push(done);
                }
            } else {
                // Runs found by binary search; each run's weighted count is a
                // prefix-sum difference — O(distinct·log occ) even when a
                // common chunk occurs in tens of thousands of distinct lines.
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
                    body.push((t, (self.wprefix[a] - self.wprefix[r]) as u32));
                    r = a;
                }
            }
        }
        // Fold the pending buffer in (small; weights read live).
        let mut acc: FxHashMap<TokenId, u32> = FxHashMap::default();
        if ctx.len() < self.buf.len() {
            for i in 0..=self.buf.len() - ctx.len() {
                if &self.buf[i..i + ctx.len()] == ctx {
                    let j = i + ctx.len();
                    if j < self.buf.len() {
                        let w =
                            Self::weight_at(&self.pending_offsets, &self.utts, self.committed, i);
                        *acc.entry(self.buf[j]).or_insert(0) += w;
                    }
                }
            }
        }
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
        if self.stats_dirty {
            self.recompute_stats();
            self.stats_dirty = false;
        }
        if self.wprefix_stale {
            self.refresh_wprefix();
        }
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

    /// Whether the exact token run occurs anywhere in the corpus.
    pub fn contains_seq(&self, pat: &[TokenId]) -> bool {
        if pat.is_empty() {
            return false;
        }
        if sa_range(&self.text, &self.sa, pat).is_some() {
            return true;
        }
        self.buf.windows(pat.len()).any(|w| w == pat)
    }

    /// Corpus count of a single token (0 if unseen), multiplicities included.
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
            if self.contains_seq(&hay[..n]) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_after_merge() {
        let mut s = Store::new(4);
        s.push_utterance(&[110, 111, 112]);
        s.push_utterance(&[110, 111, 113]);
        s.merge();
        let (c, total) = s.next_counts(&[110, 111]);
        assert!(total >= 2);
        let ids: Vec<TokenId> = c.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&112) || ids.contains(&113));
    }

    /// Incremental (buffered) stats must equal a from-scratch recompute over
    /// the same multiset, duplicates included.
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
        assert_eq!(a.len(), b.len());
    }

    /// Deferred replay pushes + one final merge must equal per-utterance
    /// pushes with periodic merges, duplicates included.
    #[test]
    fn deferred_replay_equals_incremental_pushes() {
        let mut a = Store::new(8); // tiny threshold → many merges on the way
        let mut b = Store::new(8);
        let mut utts: Vec<Vec<TokenId>> = Vec::new();
        for i in 0..60u32 {
            utts.push(vec![20 + i % 4, 30 + i % 3, 40 + i % 2]);
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
        assert_eq!(a.len(), b.len());
        assert_eq!(a.unigram_counts(), b.unigram_counts());
        assert_eq!(a.continuation_unigram(), b.continuation_unigram());
        assert_eq!(a.bigram_count_of_counts(), b.bigram_count_of_counts());
    }

    /// Flat-stream reference: the dedup store must return exactly what the
    /// old store computed over the full (repetitive) stream. This is the
    /// load-bearing equivalence test for the whole dedup design.
    #[test]
    fn dedup_store_matches_flat_stream_reference() {
        // Deterministic pseudo-random multiset with heavy repeats.
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = move || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 33) as u32
        };
        let pool: Vec<Vec<TokenId>> = (0..12)
            .map(|_| {
                let len = 1 + (rnd() % 5) as usize;
                (0..len).map(|_| 20 + rnd() % 9).collect()
            })
            .collect();
        let mut store = Store::new(32); // small threshold → mid-run merges
        let mut flat: Vec<TokenId> = Vec::new();
        let mut pushed: Vec<Vec<TokenId>> = Vec::new();
        for _ in 0..300 {
            let u = &pool[(rnd() % pool.len() as u32) as usize];
            store.push_utterance(u);
            flat.push(SEP);
            flat.extend_from_slice(u);
            flat.push(EOS);
            pushed.push(u.clone());
        }
        // leave some in the pending buffer on purpose (no final merge)

        // unigram
        let mut uni: FxHashMap<TokenId, u32> = FxHashMap::default();
        for &t in &flat {
            *uni.entry(t).or_insert(0) += 1;
        }
        let mut uni_v: Vec<(TokenId, u32)> = uni.into_iter().collect();
        uni_v.sort_by_key(|(id, _)| *id);
        assert_eq!(store.unigram_counts(), uni_v);
        assert_eq!(store.len(), flat.len());

        // bigrams / continuation / count-of-counts
        let mut big: FxHashMap<(TokenId, TokenId), u32> = FxHashMap::default();
        for w in flat.windows(2) {
            *big.entry((w[0], w[1])).or_insert(0) += 1;
        }
        let mut cont: FxHashMap<TokenId, u32> = FxHashMap::default();
        let (mut n1, mut n2, mut n3, mut n4) = (0u64, 0u64, 0u64, 0u64);
        for (&(_, w), &c) in &big {
            *cont.entry(w).or_insert(0) += 1;
            match c {
                1 => n1 += 1,
                2 => n2 += 1,
                3 => n3 += 1,
                4 => n4 += 1,
                _ => {}
            }
        }
        assert_eq!(store.bigram_count_of_counts(), (n1, n2, n3, n4));
        let tot: u32 = cont.values().sum();
        let mut cont_v: Vec<(TokenId, f64)> = cont
            .into_iter()
            .map(|(id, c)| (id, c as f64 / tot as f64))
            .collect();
        cont_v.sort_by_key(|(id, _)| *id);
        assert_eq!(store.continuation_unigram(), cont_v);

        // next_counts for many special-free contexts vs flat scan
        let mut ctxs: Vec<Vec<TokenId>> = Vec::new();
        for u in pool.iter() {
            for len in 1..=u.len().min(3) {
                for start in 0..=u.len() - len {
                    ctxs.push(u[start..start + len].to_vec());
                }
            }
        }
        ctxs.push(vec![7]); // unseen
        for ctx in &ctxs {
            let mut acc: FxHashMap<TokenId, u32> = FxHashMap::default();
            for i in 0..flat.len().saturating_sub(ctx.len()) {
                if &flat[i..i + ctx.len()] == ctx.as_slice() {
                    *acc.entry(flat[i + ctx.len()]).or_insert(0) += 1;
                }
            }
            let total: u32 = acc.values().copied().sum();
            let mut want: Vec<(TokenId, u32)> = acc.into_iter().collect();
            want.sort_by_key(|(id, _)| *id);
            let (got, got_total) = store.next_counts(ctx);
            assert_eq!(got, want, "ctx={ctx:?}");
            assert_eq!(got_total, total, "ctx={ctx:?}");
        }

        // contains_seq on windows of pushed utterances and on absent runs
        for u in pool.iter() {
            for len in 1..=u.len() {
                assert!(store.contains_seq(&u[..len]), "prefix of {u:?}");
            }
        }
        assert!(!store.contains_seq(&[7, 8, 9]));
    }

    #[test]
    fn mkn_discounts_increase_with_count_bin() {
        let mut s = Store::new(32);
        for i in 0..40u32 {
            s.push_utterance(&[100, 200 + i]);
        }
        for _ in 0..8 {
            s.push_utterance(&[110, 111]);
        }
        for _ in 0..3 {
            s.push_utterance(&[112, 113]);
        }
        for _ in 0..2 {
            s.push_utterance(&[114, 115]);
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
