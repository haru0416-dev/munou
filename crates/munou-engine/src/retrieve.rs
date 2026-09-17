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
    /// embeddings — one embed per distinct text (the log repeats lines
    /// heavily; the embedder is deterministic). Called once at the end of
    /// open / retokenize.
    pub(crate) fn heap_bytes(&self) -> usize {
        let mut b = self.items.capacity() * std::mem::size_of::<BotUtterance>();
        for it in &self.items {
            b += it.text.capacity() + it.toks.capacity() * 4 + it.emb.capacity() * 4;
        }
        b
    }

    /// Snapshot payload: the retained window without embeddings (re-embedded
    /// on load; same embedder, same values).
    pub(crate) fn snap_items(&self) -> impl Iterator<Item = (&str, &[TokenId])> + '_ {
        self.items.iter().map(|b| (b.text.as_str(), &b.toks[..]))
    }

    pub fn finish(&mut self, embedder: &HashEmbedder, scan_cap: usize) {
        if scan_cap > 0 && self.items.len() > scan_cap {
            let cut = self.items.len() - scan_cap;
            self.items.drain(..cut);
        }
        // Replay pushes every learned turn before the trim above; drain
        // keeps the full capacity (~88B per original entry) unless returned.
        self.items.shrink_to_fit();
        let mut memo: rustc_hash::FxHashMap<String, Vec<f32>> = rustc_hash::FxHashMap::default();
        for b in self.items.iter_mut() {
            b.emb = memo
                .entry(b.text.clone())
                .or_insert_with(|| {
                    let mut v = vec![0.0f32; embedder.dim()];
                    embedder.embed(&b.text, &mut v);
                    v
                })
                .clone();
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
        // Pool deduplication makes only the first occurrence of each text
        // eligible. Keep its tokens: equal text can have different historical
        // tokenization. Blank proposals are rejected by Pool but still consume
        // a pick, so leave those occurrences separate.
        let mut seen: rustc_hash::FxHashSet<&str> =
            pool.items.iter().map(|p| p.text.as_str()).collect();
        for (i, b) in self.items.iter().enumerate().skip(start) {
            if b.text == input || seen.contains(b.text.as_str()) {
                continue;
            }
            if !b.text.trim().is_empty() {
                seen.insert(b.text.as_str());
            }
            cands.push((cosine(topic, &b.emb), i));
        }
        let mut picked_flag = vec![false; cands.len()];
        let mut red = vec![0.0f32; cands.len()];
        let mut n_picked = 0usize;
        while n_picked < n_retrieve && n_picked < cands.len() {
            let mut best_i = None;
            let mut best_s = f32::NEG_INFINITY;
            for (ci, &(sim, _)) in cands.iter().enumerate() {
                if picked_flag[ci] {
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
            if n_picked == n_retrieve {
                break;
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    // Direct MMR oracle: retain every occurrence and recompute redundancy from
    // all previous picks. Compare proposals, including their historical tokens.
    fn reference(
        bots: &BotStore,
        pool: &mut Pool,
        input: &str,
        topic: &[f32],
        n: usize,
        lambda: f32,
        cap: usize,
    ) {
        let mut picked = Vec::new();
        for _ in 0..n {
            let mut best = None;
            let mut best_score = f32::NEG_INFINITY;
            for (i, b) in bots.items.iter().enumerate().skip(bots.scan_start(cap)) {
                if b.text == input
                    || picked.contains(&i)
                    || pool.items.iter().any(|p| p.text == b.text)
                {
                    continue;
                }
                let redundancy = picked
                    .iter()
                    .map(|&j| cosine(&b.emb, &bots.items[j].emb))
                    .fold(0.0f32, f32::max);
                let score = lambda * cosine(topic, &b.emb) - (1.0 - lambda) * redundancy;
                if score > best_score {
                    best_score = score;
                    best = Some(i);
                }
            }
            let Some(i) = best else { break };
            picked.push(i);
            let b = &bots.items[i];
            pool.push(PathKind::Retrieve, b.text.clone(), b.toks.clone());
        }
    }

    #[test]
    fn mmr_matches_reference_across_duplicate_and_scan_boundaries() {
        let embedder = HashEmbedder::new(64);
        let mut bots = BotStore::default();
        for (i, text) in [
            "猫かわいい",
            "散歩しよう",
            "",
            "猫かわいい",
            "コーヒー飲もう",
            " ",
            "猫かわいい",
            "",
            "散歩しよう",
            "ゲームしよう",
        ]
        .iter()
        .enumerate()
        {
            bots.push_raw((*text).into(), vec![i as TokenId + 10]);
        }
        bots.finish(&embedder, 0);
        let mut topic = vec![0.0; embedder.dim()];
        embedder.embed("猫と散歩", &mut topic);
        for cap in [0, 1, 7] {
            for lambda in [0.0, 0.75, 1.0] {
                for n in [0, 1, 4, 20] {
                    let mut actual = Pool::default();
                    let mut expected = Pool::default();
                    for pool in [&mut actual, &mut expected] {
                        pool.push(PathKind::Trigger, "コーヒー飲もう".into(), vec![99]);
                    }
                    bots.propose(&mut actual, "ゲームしよう", &topic, n, lambda, cap);
                    reference(&bots, &mut expected, "ゲームしよう", &topic, n, lambda, cap);
                    let proposals = |pool: Pool| {
                        pool.items
                            .into_iter()
                            .map(|p| (p.source, p.text, p.tokens))
                            .collect::<Vec<_>>()
                    };
                    assert_eq!(
                        proposals(actual),
                        proposals(expected),
                        "cap={cap} lambda={lambda} n={n}"
                    );
                }
            }
        }
    }
}
