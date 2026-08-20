//! Adapt source — rewrite of a learned exchange, closed over the own log.
//! Two proposals per turn at most:
//!
//! 1. **Adapted reply**: find the learned user utterance most similar to the
//!    current input, take the reply that followed it, and substitute content
//!    chunks that differ between the two inputs (「Rubyってどうよ」→
//!    「Railsってどうよ」 turns 「Rubyサイコー」 into 「Railsサイコー」).
//!    Rewriting a real exchange keeps the grammar of the original.
//! 2. **Quoted user line**: the past user utterance closest to the current
//!    topic, spoken back (role inversion). The bot remembering the user's
//!    own words is the point; no completeness.
//!
//! Deterministic, no RNG. Everything comes from learned turns of the log.

use crate::embed::{cosine, Embedder, HashEmbedder};
use crate::explain::PathKind;
use crate::ids::{is_special, TokenId};
use crate::intern::Interner;
use crate::mix::Pool;
use crate::store::Store;
use crate::tokenizer::{detokenize, is_punct_str};

#[derive(Debug, Clone)]
struct Pair {
    user_text: String,
    user_chunks: Vec<TokenId>,
    reply_chunks: Vec<TokenId>,
    /// Embedding of `user_text`, cached like the retrieve store.
    emb: Vec<f32>,
}

/// Learned (user utterance → following reply) pairs, oldest first. The same
/// scan cap as retrieve bounds the window.
#[derive(Debug, Default)]
pub(crate) struct PairStore {
    items: Vec<Pair>,
}

impl PairStore {
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Replay-time push without an embedding; `finish` fills them in bulk.
    pub fn push_raw(
        &mut self,
        user_text: String,
        user_chunks: Vec<TokenId>,
        reply_chunks: Vec<TokenId>,
    ) {
        self.items.push(Pair {
            user_text,
            user_chunks,
            reply_chunks,
            emb: Vec::new(),
        });
    }

    /// Trim to the scan window, then embed — one embed per distinct user
    /// text (see `BotStore::finish`). Called once at end of replay.
    pub fn finish(&mut self, embedder: &HashEmbedder, scan_cap: usize) {
        if scan_cap > 0 && self.items.len() > scan_cap {
            let cut = self.items.len() - scan_cap;
            self.items.drain(..cut);
        }
        let mut memo: rustc_hash::FxHashMap<String, Vec<f32>> = rustc_hash::FxHashMap::default();
        for p in self.items.iter_mut() {
            p.emb = memo
                .entry(p.user_text.clone())
                .or_insert_with(|| {
                    let mut v = vec![0.0f32; embedder.dim()];
                    embedder.embed(&p.user_text, &mut v);
                    v
                })
                .clone();
        }
    }

    /// Live push with an amortised front-trim (2×cap keeps behaviour exact,
    /// same rule as the retrieve store).
    pub fn push_live(
        &mut self,
        embedder: &HashEmbedder,
        user_text: String,
        user_chunks: Vec<TokenId>,
        reply_chunks: Vec<TokenId>,
        scan_cap: usize,
    ) {
        let mut emb = vec![0.0f32; embedder.dim()];
        embedder.embed(&user_text, &mut emb);
        self.items.push(Pair {
            user_text,
            user_chunks,
            reply_chunks,
            emb,
        });
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

    /// Propose into the pool. `input_emb` matches the utterance,
    /// `topic` picks the quoted user line (avoids plain echo of the input).
    #[allow(clippy::too_many_arguments)]
    pub fn propose(
        &self,
        pool: &mut Pool,
        intern: &Interner,
        store: &Store,
        input: &str,
        input_chunks: &[TokenId],
        input_emb: &[f32],
        topic: &[f32],
        n_adapt: usize,
        scan_cap: usize,
    ) {
        if self.items.is_empty() || n_adapt == 0 {
            return;
        }
        let start = self.scan_start(scan_cap);

        // 1. Adapted reply from the most input-similar past exchange.
        let mut best: Option<(f32, usize)> = None;
        for (i, p) in self.items.iter().enumerate().skip(start) {
            if p.user_text == input {
                continue; // identical exchange would just replay itself
            }
            let sim = cosine(input_emb, &p.emb);
            if best.map(|(s, _)| sim > s).unwrap_or(true) {
                best = Some((sim, i));
            }
        }
        if let Some((_, i)) = best {
            let base = &self.items[i];
            let toks = substitute(store, intern, base, input_chunks);
            let strs: Vec<String> = toks
                .iter()
                .copied()
                .filter(|id| !is_special(*id))
                .map(|id| intern.get(id).to_string())
                .collect();
            let text = detokenize(&strs);
            pool.push(PathKind::Adapt, text, toks);
        }

        // 2. Quoted past user line, matched on topic (the bot remembers what
        //    you said around here).
        if n_adapt >= 2 {
            let mut best_q: Option<(f32, usize)> = None;
            for (i, p) in self.items.iter().enumerate().skip(start) {
                if p.user_text == input {
                    continue;
                }
                let sim = cosine(topic, &p.emb);
                if best_q.map(|(s, _)| sim > s).unwrap_or(true) {
                    best_q = Some((sim, i));
                }
            }
            if let Some((_, i)) = best_q {
                let p = &self.items[i];
                pool.push(PathKind::Adapt, p.user_text.clone(), p.user_chunks.clone());
            }
        }
    }
}

/// Content chunk: carries surface meaning (not a special, not punctuation).
fn is_content(intern: &Interner, id: TokenId) -> bool {
    !is_special(id) && !is_punct_str(intern.get(id))
}

/// Substitution at chunk level: chunks unique to the base input that occur
/// in the base reply are replaced by chunks unique to the new input.
/// Rarest first on both sides (rarest = most contentful); at most two pairs.
/// No substitution found → the reply proposes as-is (pair-following
/// retrieval is still informative).
fn substitute(
    store: &Store,
    intern: &Interner,
    base: &Pair,
    input_chunks: &[TokenId],
) -> Vec<TokenId> {
    let absent = |id: &TokenId, other: &[TokenId]| !other.contains(id);
    let mut old_only: Vec<TokenId> = base
        .user_chunks
        .iter()
        .copied()
        .filter(|id| {
            is_content(intern, *id) && absent(id, input_chunks) && base.reply_chunks.contains(id)
        })
        .collect();
    let mut new_only: Vec<TokenId> = input_chunks
        .iter()
        .copied()
        .filter(|id| is_content(intern, *id) && absent(id, &base.user_chunks))
        .collect();
    // Full dedup (Vec::dedup only removes *adjacent* repeats, which wasted a
    // substitution slot on the same chunk and could map two old chunks onto
    // one new chunk), then rarest first.
    let mut seen = rustc_hash::FxHashSet::default();
    old_only.retain(|id| seen.insert(*id));
    seen.clear();
    new_only.retain(|id| seen.insert(*id));
    old_only.sort_by_key(|&id| store.count_of(id));
    new_only.sort_by_key(|&id| store.count_of(id));

    let mut out = base.reply_chunks.clone();
    for (o, n) in old_only.iter().zip(new_only.iter()).take(2) {
        for t in out.iter_mut() {
            if *t == *o {
                *t = *n;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::HashEmbedder;

    #[test]
    fn substitutes_differing_content_chunk() {
        let mut intern = Interner::new();
        let ruby = intern.intern("Ruby");
        let dou = intern.intern("ってどうよ");
        let saiko = intern.intern("サイコー");
        let rails = intern.intern("Rails");
        let store = Store::new(64);

        let mut ps = PairStore::default();
        ps.push_raw("Rubyってどうよ".into(), vec![ruby, dou], vec![ruby, saiko]);
        let e = HashEmbedder::new(64);
        ps.finish(&e, 0);

        let mut q = vec![0.0f32; 64];
        e.embed("Railsってどうよ", &mut q);
        let mut pool = Pool::default();
        ps.propose(
            &mut pool,
            &intern,
            &store,
            "Railsってどうよ",
            &[rails, dou],
            &q,
            &q,
            1,
            0,
        );
        assert_eq!(pool.items.len(), 1);
        assert_eq!(pool.items[0].source, PathKind::Adapt);
        assert_eq!(pool.items[0].text, "Railsサイコー", "{:?}", pool.items[0]);
    }

    /// Non-adjacent duplicates must not eat the two substitution slots
    /// (Vec::dedup only removes adjacent repeats): user [x,z,x] / reply [x,z]
    /// with input [n1,n2] must substitute both x→ and z→, not stop at x.
    #[test]
    fn substitution_survives_non_adjacent_duplicates() {
        let mut intern = Interner::new();
        let x = intern.intern("えっくす");
        let z = intern.intern("ぜっと");
        let n1 = intern.intern("いち");
        let n2 = intern.intern("に");
        let store = Store::new(64);
        let mut ps = PairStore::default();
        ps.push_raw("えっくすぜっとえっくす".into(), vec![x, z, x], vec![x, z]);
        let e = HashEmbedder::new(64);
        ps.finish(&e, 0);
        let mut q = vec![0.0f32; 64];
        e.embed("いちに", &mut q);
        let mut pool = Pool::default();
        ps.propose(
            &mut pool,
            &intern,
            &store,
            "いちに",
            &[n1, n2],
            &q,
            &q,
            1,
            0,
        );
        assert_eq!(pool.items.len(), 1);
        let toks = &pool.items[0].tokens;
        assert!(
            toks.contains(&n1) && toks.contains(&n2),
            "both differing chunks must be substituted: {toks:?}"
        );
    }

    #[test]
    fn quotes_past_user_line_as_second_proposal() {
        let mut intern = Interner::new();
        let neko = intern.intern("猫かわいい");
        let hai = intern.intern("そうだね");
        let store = Store::new(64);
        let mut ps = PairStore::default();
        ps.push_raw("猫かわいい".into(), vec![neko], vec![hai]);
        let e = HashEmbedder::new(64);
        ps.finish(&e, 0);
        let mut q = vec![0.0f32; 64];
        e.embed("うちの猫の話", &mut q);
        let mut pool = Pool::default();
        ps.propose(
            &mut pool,
            &intern,
            &store,
            "うちの猫の話",
            &[],
            &q,
            &q,
            2,
            0,
        );
        assert!(
            pool.items.iter().any(|p| p.text == "猫かわいい"),
            "past user line should be quoted: {:?}",
            pool.items
        );
    }
}
