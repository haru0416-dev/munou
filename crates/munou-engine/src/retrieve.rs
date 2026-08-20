//! Retrieve source: past bot utterances proposed via MMR over hash embeddings.
//! One of the four pool sources (trigger / retrieve / Markov / echo).

use crate::embed::{cosine, Embedder, HashEmbedder};
use crate::explain::PathKind;
use crate::ids::TokenId;
use crate::mix::Pool;

#[derive(Debug, Clone)]
struct BotUtterance {
    text: String,
    toks: Vec<TokenId>,
    /// Hash embedding of `text`, cached so retrieve / routing do not re-embed
    /// the scan window every turn. Same embedder, same values.
    emb: Vec<f32>,
}

/// Past bot utterances, oldest first. Only the last `scan_cap` entries are
/// ever scanned (`scan_cap == 0` scans everything), so the front is trimmed.
#[derive(Debug, Default)]
pub(crate) struct BotStore {
    items: Vec<BotUtterance>,
}

impl BotStore {
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Replay-time push without an embedding; `finish` fills them in bulk.
    pub fn push_raw(&mut self, text: String, toks: Vec<TokenId>) {
        self.items.push(BotUtterance {
            text,
            toks,
            emb: Vec::new(),
        });
    }

    /// Drop entries that can never enter the scan window again, then fill
    /// embeddings. Called once at the end of open / retokenize.
    pub fn finish(&mut self, embedder: &HashEmbedder, scan_cap: usize) {
        if scan_cap > 0 && self.items.len() > scan_cap {
            let cut = self.items.len() - scan_cap;
            self.items.drain(..cut);
        }
        for b in self.items.iter_mut() {
            let mut v = vec![0.0f32; embedder.dim()];
            embedder.embed(&b.text, &mut v);
            b.emb = v;
        }
    }

    /// Live push with an embedding, plus an amortised front-trim: entries
    /// before the last `scan_cap` are unreachable, so keeping at most 2·cap
    /// preserves behaviour exactly.
    pub fn push_live(
        &mut self,
        embedder: &HashEmbedder,
        text: String,
        toks: Vec<TokenId>,
        scan_cap: usize,
    ) {
        let mut emb = vec![0.0f32; embedder.dim()];
        embedder.embed(&text, &mut emb);
        self.items.push(BotUtterance { text, toks, emb });
        if scan_cap > 0 && self.items.len() > scan_cap * 2 {
            let cut = self.items.len() - scan_cap;
            self.items.drain(..cut);
        }
    }

    fn scan_start(&self, scan_cap: usize) -> usize {
        let n = self.items.len();
        if scan_cap == 0 || scan_cap >= n {
            0
        } else {
            n - scan_cap
        }
    }

    /// Max topic cosine over the scan window (route gate input).
    pub fn max_sim(&self, topic: &[f32], scan_cap: usize) -> f32 {
        if self.items.is_empty() {
            return 0.0;
        }
        let mut m = 0.0f32;
        for b in self.items.iter().skip(self.scan_start(scan_cap)) {
            m = m.max(cosine(topic, &b.emb));
        }
        m
    }

    /// Propose up to `n_retrieve` utterances into the pool with MMR:
    /// `λ·sim − (1−λ)·max redundancy`. The max-redundancy per candidate is
    /// maintained incrementally — same value as a per-round fold, updated in
    /// O(n) per pick instead of recomputed in O(n·k) cosines per round.
    pub fn propose(
        &self,
        pool: &mut Pool,
        input: &str,
        topic: &[f32],
        n_retrieve: usize,
        mmr_lambda: f32,
        scan_cap: usize,
    ) {
        if self.items.is_empty() || n_retrieve == 0 {
            return;
        }
        let lambda = mmr_lambda.clamp(0.0, 1.0);
        let start = self.scan_start(scan_cap);
        let mut cands: Vec<(f32, usize)> = Vec::with_capacity(self.items.len() - start);
        for (i, b) in self.items.iter().enumerate().skip(start) {
            if b.text == input {
                continue;
            }
            cands.push((cosine(topic, &b.emb), i));
        }
        let mut picked_flag = vec![false; cands.len()];
        let mut red = vec![0.0f32; cands.len()];
        let mut n_picked = 0usize;
        while n_picked < n_retrieve && n_picked < cands.len() {
            let mut best_i = None;
            let mut best_s = f32::NEG_INFINITY;
            for (ci, &(sim, bot_i)) in cands.iter().enumerate() {
                if picked_flag[ci] {
                    continue;
                }
                if pool.items.iter().any(|p| p.text == self.items[bot_i].text) {
                    continue;
                }
                let mmr = lambda * sim - (1.0 - lambda) * red[ci];
                if mmr > best_s {
                    best_s = mmr;
                    best_i = Some(ci);
                }
            }
            let Some(ci) = best_i else {
                break;
            };
            picked_flag[ci] = true;
            n_picked += 1;
            let bot_i = cands[ci].1;
            let b = &self.items[bot_i];
            pool.push(PathKind::Retrieve, b.text.clone(), b.toks.clone());
            let picked_emb = &self.items[bot_i].emb;
            for (cj, &(_, bj)) in cands.iter().enumerate() {
                if !picked_flag[cj] {
                    let r = cosine(&self.items[bj].emb, picked_emb);
                    if r > red[cj] {
                        red[cj] = r;
                    }
                }
            }
        }
    }
}
