//! Candidate ranking by topic cosine, with intentional slip injection.

use rand::Rng;

use crate::embed::{cosine, Embedder};
use crate::explain::CandidateTrace;
use crate::ids::TokenId;
use crate::params::Params;

#[derive(Debug, Clone)]
pub struct Ranked {
    pub index: usize,
    pub slipped: bool,
    pub slip_roll: f64,
    pub traces: Vec<CandidateTrace>,
}

pub fn rank_and_pick<R: Rng + ?Sized, E: Embedder>(
    embedder: &E,
    topic: &[f32],
    texts: &[String],
    tokens: &[Vec<TokenId>],
    params: &Params,
    rng: &mut R,
) -> Ranked {
    let dim = embedder.dim();
    let mut buf = vec![0.0f32; dim];
    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(texts.len());
    for (i, t) in texts.iter().enumerate() {
        embedder.embed(t, &mut buf);
        scored.push((i, cosine(topic, &buf)));
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let slip_roll: f64 = rng.gen();
    let slipped = slip_roll < params.p_slip && scored.len() >= 2;
    let chosen_rank = if slipped {
        // sample among 2nd..last, weighted by shifted cosine
        let rest = &scored[1..];
        let mut w: Vec<f64> = rest
            .iter()
            .map(|(_, s)| (*s as f64 + 1.0).max(1e-6))
            .collect();
        let z: f64 = w.iter().sum();
        if z > 0.0 {
            for x in w.iter_mut() {
                *x /= z;
            }
        }
        let table = crate::alias::AliasTable::from_weights(&w);
        1 + table.sample(rng)
    } else {
        0
    };
    let index = scored.get(chosen_rank).map(|p| p.0).unwrap_or(0);

    let traces = scored
        .iter()
        .enumerate()
        .map(|(rank, (i, score))| CandidateTrace {
            rank,
            text: texts[*i].clone(),
            tokens: tokens.get(*i).cloned().unwrap_or_default(),
            score: *score,
            chosen: *i == index,
        })
        .collect();

    Ranked {
        index,
        slipped,
        slip_roll,
        traces,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::HashEmbedder;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn top_rank_without_slip() {
        let e = HashEmbedder::new(64);
        let mut topic = vec![0.0; 64];
        e.embed("天気は晴れ", &mut topic);
        let texts = vec!["今日は晴れだね".into(), "xml hashing".into()];
        let toks = vec![vec![1], vec![2]];
        let params = Params {
            p_slip: 0.0,
            ..Params::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let r = rank_and_pick(&e, &topic, &texts, &toks, &params, &mut rng);
        assert!(!r.slipped);
        assert_eq!(r.traces[0].text, "今日は晴れだね");
    }
}
