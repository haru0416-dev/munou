//! Conversation corpus: a single `u32` stream + suffix array + generation buffer.
//!
//! Variable-length n-grams are looked up by binary search on the SA (body)
//! and by a linear scan of the small generation buffer. When the buffer
//! exceeds `merge_threshold`, the body is rebuilt with SA-IS.

use rustc_hash::FxHashMap;

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
    /// Bigram count-of-counts n1..n4 for Chen-Goodman modified KN.
    bigram_n1: u64,
    bigram_n2: u64,
    bigram_n3: u64,
    bigram_n4: u64,
    merge_threshold: usize,
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
        }
    }

    fn bump_unigram_slice(&mut self, slice: &[TokenId]) {
        for &t in slice {
            *self.unigram.entry(t).or_insert(0) += 1;
        }
    }

    pub fn merge(&mut self) {
        if self.buf.is_empty() && !self.text.is_empty() && !self.sa.is_empty() {
            return;
        }
        self.text.extend_from_slice(&self.buf);
        self.buf.clear();
        self.rebuild_index();
    }

    pub fn rebuild_index(&mut self) {
        self.sa = suffix_array(&self.text);
        self.recompute_stats();
    }

    fn recompute_stats(&mut self) {
        self.unigram.clear();
        self.continuation.clear();
        self.bigram_n1 = 0;
        self.bigram_n2 = 0;
        self.bigram_n3 = 0;
        self.bigram_n4 = 0;
        for &t in self.text.iter().chain(self.buf.iter()) {
            *self.unigram.entry(t).or_insert(0) += 1;
        }
        // unique left contexts of each token in the body
        let mut seen: FxHashMap<TokenId, FxHashMap<TokenId, ()>> = FxHashMap::default();
        let stream: Vec<TokenId> = self.text.iter().chain(self.buf.iter()).copied().collect();
        let mut bigrams: FxHashMap<(TokenId, TokenId), u32> = FxHashMap::default();
        for i in 1..stream.len() {
            let left = stream[i - 1];
            let w = stream[i];
            seen.entry(w).or_default().insert(left, ());
            *bigrams.entry((left, w)).or_insert(0) += 1;
        }
        self.continuation = seen
            .into_iter()
            .map(|(w, lefts)| (w, lefts.len() as u32))
            .collect();
        for &c in bigrams.values() {
            match c {
                1 => self.bigram_n1 += 1,
                2 => self.bigram_n2 += 1,
                3 => self.bigram_n3 += 1,
                4 => self.bigram_n4 += 1,
                _ => {}
            }
        }
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
    pub fn next_counts(&self, ctx: &[TokenId]) -> (Vec<(TokenId, u32)>, u32) {
        let mut acc: FxHashMap<TokenId, u32> = FxHashMap::default();
        if !ctx.is_empty() {
            if let Some((lo, hi)) = sa_range(&self.text, &self.sa, ctx) {
                for r in lo..hi {
                    let pos = sa_pos(self.sa[r], ctx.len(), self.text.len());
                    if let Some(tok) = pos.and_then(|p| self.text.get(p).copied()) {
                        *acc.entry(tok).or_insert(0) += 1;
                    }
                }
            }
            count_linear(&self.buf, ctx, &mut acc);
        }
        let total: u32 = acc.values().copied().sum();
        let mut v: Vec<(TokenId, u32)> = acc.into_iter().collect();
        v.sort_by_key(|(id, _)| *id);
        (v, total)
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

fn occurs(store: &Store, pat: &[TokenId]) -> bool {
    if sa_range(&store.text, &store.sa, pat).is_some() {
        return true;
    }
    store.buf.windows(pat.len()).any(|w| w == pat)
}

fn sa_pos(sa_i: u32, ctx_len: usize, n: usize) -> Option<usize> {
    let p = sa_i as usize + ctx_len;
    if p < n {
        Some(p)
    } else {
        None
    }
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
