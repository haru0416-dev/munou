//! Markov generation: interpolated variable-order n-grams over a suffix array.

use rand::Rng;

use crate::alias::{temper, AliasTable};
use crate::explain::GenStep;
use crate::ids::{TokenId, EOS};
use crate::params::Params;
use crate::smoothing::Smoothing;
use crate::store::Store;

#[derive(Debug, Clone)]
pub struct Generated {
    pub tokens: Vec<TokenId>,
    pub steps: Vec<GenStep>,
}

pub fn generate_one<R: Rng + ?Sized>(
    store: &Store,
    smoothing: &dyn Smoothing,
    params: &Params,
    ctx_seed: &[TokenId],
    parrot: &[TokenId],
    rng: &mut R,
) -> Generated {
    let mut tokens: Vec<TokenId> = Vec::new();
    let mut steps: Vec<GenStep> = Vec::new();
    let l_max = params.l_max_capped();

    let cap = if store.is_empty() {
        parrot.len().clamp(1, 8)
    } else {
        params.max_gen_len
    };

    for _ in 0..cap {
        let mut ctx: Vec<TokenId> = Vec::new();
        ctx.extend_from_slice(ctx_seed);
        ctx.extend_from_slice(&tokens);
        if ctx.len() > l_max {
            let skip = ctx.len() - l_max;
            ctx.drain(..skip);
        }

        let requested = ctx.len();
        let (ids, weights, used_len, freq) =
            dist_with_backoff(store, smoothing, params, &ctx, parrot);

        if ids.is_empty() {
            break;
        }

        let mut w = weights;
        temper(&mut w, params.tau_gen);
        let table = AliasTable::from_weights(&w);
        let idx = table.sample(rng);
        let sampled = ids[idx];

        steps.push(GenStep {
            ctx_len_requested: requested,
            ctx_len_used: used_len,
            freq,
            sampled,
            temperature: params.tau_gen as f32,
        });

        if sampled == EOS {
            break;
        }
        tokens.push(sampled);
    }

    if tokens.is_empty() {
        if let Some(&t) = parrot.first() {
            tokens.push(t);
            steps.push(GenStep {
                ctx_len_requested: 0,
                ctx_len_used: 0,
                freq: 0,
                sampled: t,
                temperature: params.tau_gen as f32,
            });
        }
    }

    Generated { tokens, steps }
}

fn dist_with_backoff(
    store: &Store,
    smoothing: &dyn Smoothing,
    params: &Params,
    ctx: &[TokenId],
    parrot: &[TokenId],
) -> (Vec<TokenId>, Vec<f64>, usize, u32) {
    let unigram = if matches!(params.smoothing, crate::params::SmoothingKind::Kn) {
        store.continuation_unigram()
    } else {
        store.ml_unigram()
    };
    let mut backoff = if unigram.is_empty() {
        parrot_unigram(parrot)
    } else {
        unigram
    };

    // Interpolate every non-empty order, shortest suffix first.
    // P_n = mix(counts(ctx[-n:]), P_{n-1}). Skipping to unigram was the
    // previous bug: a long match threw away the intermediate n-grams.
    let mut used = 0usize;
    let mut freq = 0u32;
    for len in 1..=ctx.len() {
        let sub = &ctx[ctx.len() - len..];
        let (counts, total) = store.next_counts(sub);
        if total == 0 {
            continue;
        }
        let n1plus = counts.len() as u32;
        backoff = smoothing.distribute(&counts, total, n1plus, &backoff);
        if total >= params.f_min || freq < params.f_min {
            used = len;
            freq = total;
        }
    }

    if backoff.is_empty() {
        backoff = parrot_unigram(parrot);
    }
    if backoff.is_empty() {
        return (Vec::new(), Vec::new(), used, freq);
    }
    let ids: Vec<TokenId> = backoff.iter().map(|(id, _)| *id).collect();
    let w: Vec<f64> = backoff.iter().map(|(_, p)| *p).collect();
    (ids, w, used, freq)
}

fn parrot_unigram(parrot: &[TokenId]) -> Vec<(TokenId, f64)> {
    if parrot.is_empty() {
        return Vec::new();
    }
    let mut counts: rustc_hash::FxHashMap<TokenId, u32> = rustc_hash::FxHashMap::default();
    let mut order: Vec<TokenId> = Vec::new();
    for &t in parrot {
        let e = counts.entry(t).or_insert(0);
        if *e == 0 {
            order.push(t);
        }
        *e += 1;
    }
    let inv = 1.0 / parrot.len() as f64;
    order
        .into_iter()
        .map(|id| (id, counts[&id] as f64 * inv))
        .collect()
}

/// Longest common token *subsequence* (can skip). Kept for tests / contrast.
pub fn lcs_len(a: &[TokenId], b: &[TokenId]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let (x, y) = if a.len() < b.len() { (a, b) } else { (b, a) };
    let mut prev = vec![0usize; x.len() + 1];
    let mut cur = vec![0usize; x.len() + 1];
    for &by in y {
        for (i, &ax) in x.iter().enumerate() {
            cur[i + 1] = if ax == by {
                prev[i] + 1
            } else {
                prev[i + 1].max(cur[i])
            };
        }
        std::mem::swap(&mut prev, &mut cur);
        cur.fill(0);
    }
    prev[x.len()]
}

/// Longest common *contiguous* token run. This is the design's 最長一致
/// (rote copy). Subsequence LCS over-penalises Markov recombination.
pub fn lcsubstr_len(a: &[TokenId], b: &[TokenId]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let (x, y) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let mut prev = vec![0usize; x.len() + 1];
    let mut cur = vec![0usize; x.len() + 1];
    let mut best = 0usize;
    for &by in y {
        for (i, &ax) in x.iter().enumerate() {
            cur[i + 1] = if ax == by { prev[i] + 1 } else { 0 };
            if cur[i + 1] > best {
                best = cur[i + 1];
            }
        }
        std::mem::swap(&mut prev, &mut cur);
        cur.fill(0);
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smoothing::NaiveBackoff;
    use crate::store::Store;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn lcs_simple() {
        assert_eq!(lcs_len(&[1, 2, 3], &[2, 3, 4]), 2);
        assert_eq!(lcs_len(&[1], &[2]), 0);
    }

    #[test]
    fn lcsubstr_is_contiguous_not_subsequence() {
        assert_eq!(lcsubstr_len(&[1, 2, 3, 4], &[1, 3, 4]), 2);
        assert_eq!(lcs_len(&[1, 2, 3, 4], &[1, 3, 4]), 3);
        assert_eq!(lcsubstr_len(&[1, 2, 3], &[1, 2, 3]), 3);
        assert_eq!(lcsubstr_len(&[7], &[8]), 0);
    }

    #[test]
    fn longest_match_continues_observed_ngram() {
        let mut store = Store::new(32);
        let a = 20;
        let b = 21;
        let c = 22;
        for _ in 0..8 {
            store.push_utterance(&[a, b, c]);
        }
        store.merge();
        let params = Params {
            f_min: 3,
            l_max: 8,
            max_gen_len: 1,
            tau_gen: 1.0,
            ..Params::default()
        };
        let smoothing = NaiveBackoff;
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut hit = 0;
        for _ in 0..40 {
            let g = generate_one(&store, &smoothing, &params, &[a, b], &[], &mut rng);
            if g.tokens.first() == Some(&c) {
                hit += 1;
            }
        }
        assert!(
            hit >= 36,
            "expected continuation 黄 after 赤青; hit={hit}/40"
        );
        let g = generate_one(&store, &smoothing, &params, &[a, b], &[], &mut rng);
        assert!(!g.steps.is_empty());
        assert!(g.steps[0].ctx_len_used >= 2);
        assert!(g.steps[0].freq >= 3);
    }

    #[test]
    fn interpolation_leaks_mass_from_shorter_context() {
        // [a,x] is common; [b,a,y] is the only long match. Unigram of x is
        // drowned by many [z]. Recursive mix should still see P(x|a) ≈ 1
        // and leak x into P(·|b,a). Jumping to unigram would not.
        let mut store = Store::new(32);
        let (a, x, b, y, z) = (20u32, 21, 22, 23, 24);
        for _ in 0..20 {
            store.push_utterance(&[a, x]);
        }
        for _ in 0..3 {
            store.push_utterance(&[b, a, y]);
        }
        for _ in 0..80 {
            store.push_utterance(&[z, z]);
        }
        store.merge();
        let params = Params {
            f_min: 3,
            l_max: 8,
            max_gen_len: 1,
            tau_gen: 1.0,
            ..Params::default()
        };
        let smoothing = NaiveBackoff;
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let mut hit_x = 0;
        for _ in 0..200 {
            let g = generate_one(&store, &smoothing, &params, &[b, a], &[], &mut rng);
            if g.tokens.first() == Some(&x) {
                hit_x += 1;
            }
        }
        assert!(
            hit_x >= 8,
            "interpolated P(x|b,a) should leak from P(x|a); hit_x={hit_x}/200"
        );
    }
}
