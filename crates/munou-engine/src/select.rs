//! Candidate ranking: topic cosine + source priors − contiguous rote − band hinge.
//!
//! Maximising cosine alone fights the design's band objective (§6). In-band
//! candidates keep their cosine order; a hinge penalises sim < lo or sim > hi.
//! Slip samples 2nd+ with a Boltzmann distribution (entropy-regularised argmax).

use rand_core::Rng;

use crate::alias::AliasTable;
use crate::embed::{cosine, Embedder};
use crate::explain::{CandidateTrace, PathKind};
use crate::generate::lcsubstr_len;
use crate::ids::TokenId;
use crate::params::Params;

#[derive(Debug, Clone)]
pub struct Ranked {
    pub index: usize,
    pub slipped: bool,
    pub slip_roll: f64,
    pub traces: Vec<CandidateTrace>,
}

pub struct RankInput<'a> {
    pub topic: &'a [f32],
    pub texts: &'a [String],
    pub tokens: &'a [Vec<TokenId>],
    pub sources: &'a [PathKind],
    pub input_tokens: &'a [TokenId],
    /// Recent own reply texts (newest last); candidates repeating them lose
    /// score. Char-level on purpose: token ids drift across reopen
    /// (re-tokenised text vs generation chunks), text does not.
    pub recent_bot: &'a [String],
    /// Per-candidate mean −ln p of generation (None for non-generated).
    pub surprises: &'a [Option<f32>],
    /// Pattern-input cosine of the trigger hit, or 0.
    pub trigger_match: f32,
    /// Closed analog of RLHF: additive logit on PathKind from `/good` `/bad`.
    pub path_prior: [f32; 5],
    /// Per-candidate additive term computed upstream (関心 + 気になる語).
    /// Empty slice = no bonus. Indexed like `texts`.
    pub bonus: &'a [f32],
}

pub fn source_bias(source: PathKind, params: &Params, trigger_match: f32) -> f32 {
    match source {
        PathKind::Trigger => params.trigger_bonus + params.trigger_match_weight * trigger_match,
        PathKind::Markov => 0.0,
        PathKind::Retrieve => -params.retrieve_penalty,
        PathKind::Echo => -params.echo_penalty,
        PathKind::Adapt => -params.adapt_penalty,
    }
}

/// Longest shared contiguous run over the shorter side's length. 1.0 means
/// one side contains the other whole. Works on token ids or chars-as-u32.
fn overlap_ratio(a: &[u32], b: &[u32]) -> f32 {
    let denom = a.len().min(b.len()).max(1) as f32;
    lcsubstr_len(a, b) as f32 / denom
}

fn chars_u32(s: &str) -> Vec<u32> {
    s.chars().map(|c| c as u32).collect()
}

/// Hinge loss against the similarity band. Zero inside `[lo, hi]`.
pub fn band_hinge(sim: f32, lo: f32, hi: f32) -> f32 {
    if sim > hi {
        sim - hi
    } else if sim < lo {
        lo - sim
    } else {
        0.0
    }
}

pub fn rank_and_pick<R: Rng + ?Sized, E: Embedder>(
    embedder: &E,
    input: RankInput<'_>,
    params: &Params,
    rng: &mut R,
) -> Ranked {
    let dim = embedder.dim();
    let mut buf = vec![0.0f32; dim];
    let recent_chars: Vec<Vec<u32>> = input.recent_bot.iter().map(|t| chars_u32(t)).collect();
    let mut scored: Vec<(usize, f32, f32)> = Vec::with_capacity(input.texts.len());
    for (i, t) in input.texts.iter().enumerate() {
        embedder.embed(t, &mut buf);
        let topic_s = cosine(input.topic, &buf);
        let source = input.sources.get(i).copied().unwrap_or(PathKind::Markov);
        let toks = input.tokens.get(i).map(|s| s.as_slice()).unwrap_or(&[]);
        // min-length denominator: a long candidate that swallows the whole
        // input whole is a full copy, not a diluted one.
        let rote = params.rote_penalty * overlap_ratio(toks, input.input_tokens);
        let cand_chars = chars_u32(t);
        let self_rote = params.self_penalty
            * recent_chars
                .iter()
                .map(|prev| overlap_ratio(&cand_chars, prev))
                .fold(0.0f32, f32::max);
        // Surprise term, off by default (weight 0). Only generated
        // candidates carry a value, so nonzero weights bias between sources.
        let surprise_term =
            params.surprise_weight * input.surprises.get(i).copied().flatten().unwrap_or(0.0);
        let score = topic_s
            + source_bias(source, params, input.trigger_match)
            + input.path_prior[crate::route::prior_index(source)]
            + surprise_term
            + input.bonus.get(i).copied().unwrap_or(0.0)
            - rote
            - self_rote
            - params.band_penalty * band_hinge(topic_s, params.band_lo, params.band_hi);
        scored.push((i, score, topic_s));
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let slip_roll: f64 = crate::rng::rand_f64(rng);
    let slipped = slip_roll < params.p_slip && scored.len() >= 2;
    let chosen_rank = if slipped {
        let rest: Vec<f64> = scored[1..].iter().map(|(_, s, _)| *s as f64).collect();
        1 + AliasTable::softmax_sample(&rest, params.tau_slip, rng)
    } else {
        0
    };
    let index = scored.get(chosen_rank).map(|p| p.0).unwrap_or(0);

    let traces = scored
        .iter()
        .enumerate()
        .map(|(rank, (i, score, topic_s))| CandidateTrace {
            rank,
            source: input.sources.get(*i).copied().unwrap_or(PathKind::Markov),
            text: input.texts[*i].clone(),
            tokens: input.tokens.get(*i).cloned().unwrap_or_default(),
            topic_score: *topic_s,
            score: *score,
            chosen: *i == index,
            surprise: input.surprises.get(*i).copied().flatten(),
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
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    #[test]
    fn top_rank_without_slip() {
        let e = HashEmbedder::new(64);
        let mut topic = vec![0.0; 64];
        e.embed("天気は晴れ", &mut topic);
        let texts = vec!["今日は晴れだね".into(), "xml hashing".into()];
        let toks = vec![vec![1], vec![2]];
        let sources = vec![PathKind::Markov, PathKind::Markov];
        let params = Params {
            p_slip: 0.0,
            rote_penalty: 0.0,
            ..Params::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let r = rank_and_pick(
            &e,
            RankInput {
                topic: &topic,
                texts: &texts,
                tokens: &toks,
                sources: &sources,
                input_tokens: &[],
                surprises: &[],
                recent_bot: &[],
                trigger_match: 0.0,
                path_prior: [0.0; 5],
                bonus: &[],
            },
            &params,
            &mut rng,
        );
        assert!(!r.slipped);
        assert_eq!(r.traces[0].text, "今日は晴れだね");
    }

    #[test]
    fn echo_loses_to_on_topic_despite_copying_input() {
        let e = HashEmbedder::new(64);
        let mut topic = vec![0.0; 64];
        e.embed("おはよう", &mut topic);
        let texts = vec!["おはよう".into(), "おはよ・テスト応答".into()];
        let toks = vec![vec![7, 8], vec![9]];
        let sources = vec![PathKind::Echo, PathKind::Trigger];
        let params = Params {
            p_slip: 0.0,
            ..Params::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let r = rank_and_pick(
            &e,
            RankInput {
                topic: &topic,
                texts: &texts,
                tokens: &toks,
                sources: &sources,
                input_tokens: &[7, 8],
                surprises: &[],
                recent_bot: &[],
                trigger_match: 1.0,
                path_prior: [0.0; 5],
                bonus: &[],
            },
            &params,
            &mut rng,
        );
        assert_eq!(
            r.traces.iter().find(|c| c.chosen).unwrap().source,
            PathKind::Trigger
        );
    }

    /// A candidate identical to a recent own reply must lose to a fresh one
    /// of comparable topic score — the lock-in loop breaker.
    #[test]
    fn repeating_own_recent_reply_loses() {
        let e = HashEmbedder::new(64);
        let mut topic = vec![0.0; 64];
        e.embed("散歩の話", &mut topic);
        let texts = vec!["散歩しよう".into(), "公園まで歩こうか".into()];
        let toks: Vec<Vec<u32>> = vec![vec![21], vec![22, 23]];
        let sources = vec![PathKind::Markov, PathKind::Markov];
        let params = Params {
            p_slip: 0.0,
            ..Params::default()
        };
        let recent: Vec<String> = vec!["散歩しよう".into()]; // just said it
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let r = rank_and_pick(
            &e,
            RankInput {
                topic: &topic,
                texts: &texts,
                tokens: &toks,
                sources: &sources,
                input_tokens: &[],
                surprises: &[],
                recent_bot: &recent,
                trigger_match: 0.0,
                path_prior: [0.0; 5],
                bonus: &[],
            },
            &params,
            &mut rng,
        );
        let chosen = r.traces.iter().find(|c| c.chosen).unwrap();
        assert_eq!(
            chosen.text, "公園まで歩こうか",
            "exact repeat of a recent reply must not win"
        );
    }

    /// The rote denominator is min(len): padding a full copy of the input
    /// with extra tokens must not dilute the penalty below the plain copy's.
    #[test]
    fn rote_penalty_not_diluted_by_padding() {
        assert!((overlap_ratio(&[7, 8], &[7, 8]) - 1.0).abs() < 1e-6);
        assert!(
            (overlap_ratio(&[7, 8, 9, 10, 11, 12], &[7, 8]) - 1.0).abs() < 1e-6,
            "long candidate containing the whole input is still a full copy"
        );
    }

    #[test]
    fn band_hinge_zero_inside_and_positive_outside() {
        assert_eq!(band_hinge(0.5, 0.25, 0.85), 0.0);
        assert!((band_hinge(1.0, 0.25, 0.85) - 0.15).abs() < 1e-6);
        assert!((band_hinge(0.0, 0.25, 0.85) - 0.25).abs() < 1e-6);
    }
}
