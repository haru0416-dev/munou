//! Markov generation: interpolated variable-order n-grams over a suffix array.

use rand::Rng;

use crate::alias::{nucleus, temper, AliasTable};
use crate::explain::GenStep;
use crate::ids::{TokenId, EOS};
use crate::params::Params;
use crate::smoothing::Smoothing;
use crate::store::Store;
use rustc_hash::FxHashMap;

type NextMemo = FxHashMap<Vec<TokenId>, (Vec<(TokenId, u32)>, u32)>;

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
    let mut memo: NextMemo = FxHashMap::default();

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
        let (mut ids, mut weights, used_len, freq) =
            dist_with_backoff(store, smoothing, params, &ctx, parrot, &mut memo);

        if ids.is_empty() {
            break;
        }

        temper(&mut weights, params.tau_gen);
        nucleus(&mut ids, &mut weights, params.p_nucleus, params.k_top);
        let z: f64 = weights.iter().copied().filter(|x| *x > 0.0).sum();
        let table = AliasTable::from_weights(&weights);
        let idx = table.sample(rng);
        let sampled = ids[idx];
        let p = if z > 0.0 {
            (weights[idx].max(0.0) / z) as f32
        } else {
            0.0
        };
        let logp = if p > 0.0 { p.ln() } else { f32::NEG_INFINITY };

        steps.push(GenStep {
            ctx_len_requested: requested,
            ctx_len_used: used_len,
            freq,
            sampled,
            temperature: params.tau_gen as f32,
            p,
            logp,
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
                p: 1.0,
                logp: 0.0,
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
    memo: &mut NextMemo,
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
    let exclude =
        params.ppm_exclude || matches!(params.smoothing, crate::params::SmoothingKind::Kn);
    for len in 1..=ctx.len() {
        let sub = &ctx[ctx.len() - len..];
        let (counts, total) = lookup_counts(store, memo, sub);
        if total == 0 {
            continue;
        }
        let n1plus = counts.len() as u32;
        let lower = if exclude {
            exclude_seen(&backoff, &counts)
        } else {
            backoff
        };
        backoff = smoothing.distribute(&counts, total, n1plus, &lower);
        if total >= params.f_min || freq < params.f_min {
            used = len;
            freq = total;
        }
    }

    // Skip-gram + recency cache only when the contiguous suffix is missing or rare.
    // A solid longest match (the 36/40 continuation test) must not leak into these.
    let sparse = used < ctx.len() || freq < params.f_min;
    if sparse && params.lambda_skip > 0.0 && ctx.len() >= 3 {
        let mut skip = ctx.to_vec();
        skip.remove(skip.len() - 2);
        let (counts, total) = lookup_counts(store, memo, &skip);
        if total > 0 {
            let extra = counts_to_ml(&counts, total);
            backoff = mix_lambda(&backoff, &extra, params.lambda_skip);
        }
    }
    if sparse && params.lambda_cache > 0.0 && !ctx.is_empty() {
        let hist = if ctx.len() > 12 {
            &ctx[ctx.len() - 12..]
        } else {
            ctx
        };
        let extra = parrot_unigram(hist);
        backoff = mix_lambda(&backoff, &extra, params.lambda_cache);
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

fn lookup_counts(
    store: &Store,
    memo: &mut NextMemo,
    ctx: &[TokenId],
) -> (Vec<(TokenId, u32)>, u32) {
    let key = ctx.to_vec();
    if let Some(hit) = memo.get(&key) {
        hit.clone()
    } else {
        let got = store.next_counts(ctx);
        memo.insert(key, got.clone());
        got
    }
}

/// PPM-C: types already seen at this order are removed from the backoff and
/// the leftover is renormalised. Interpolation (short→long) stays; only the
/// lower-order support is trimmed.
fn exclude_seen(backoff: &[(TokenId, f64)], local: &[(TokenId, u32)]) -> Vec<(TokenId, f64)> {
    let seen: rustc_hash::FxHashSet<TokenId> = local.iter().map(|(id, _)| *id).collect();
    let mut out: Vec<(TokenId, f64)> = backoff
        .iter()
        .copied()
        .filter(|(id, p)| *p > 0.0 && !seen.contains(id))
        .collect();
    let z: f64 = out.iter().map(|(_, p)| *p).sum();
    if z > 0.0 {
        for (_, p) in out.iter_mut() {
            *p /= z;
        }
    }
    out
}

fn counts_to_ml(counts: &[(TokenId, u32)], total: u32) -> Vec<(TokenId, f64)> {
    if total == 0 {
        return Vec::new();
    }
    let t = total as f64;
    counts.iter().map(|(id, c)| (*id, *c as f64 / t)).collect()
}

fn mix_lambda(base: &[(TokenId, f64)], extra: &[(TokenId, f64)], lam: f64) -> Vec<(TokenId, f64)> {
    let lam = lam.clamp(0.0, 1.0);
    if lam <= 0.0 || extra.is_empty() {
        return base.to_vec();
    }
    let keep = 1.0 - lam;
    let mut map: FxHashMap<TokenId, f64> = FxHashMap::default();
    let mut order: Vec<TokenId> = Vec::new();
    for &(id, p) in base {
        if keep > 0.0 {
            if !map.contains_key(&id) {
                order.push(id);
            }
            *map.entry(id).or_insert(0.0) += keep * p;
        }
    }
    for &(id, p) in extra {
        if !map.contains_key(&id) {
            order.push(id);
        }
        *map.entry(id).or_insert(0.0) += lam * p;
    }
    let mut out: Vec<(TokenId, f64)> = order
        .into_iter()
        .filter_map(|id| {
            let p = map.get(&id).copied().unwrap_or(0.0);
            (p > 0.0).then_some((id, p))
        })
        .collect();
    let z: f64 = out.iter().map(|(_, p)| *p).sum();
    if z > 0.0 {
        for (_, p) in out.iter_mut() {
            *p /= z;
        }
    }
    out
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
        assert!(g.steps[0].p > 0.0 && g.steps[0].p <= 1.0);
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

    #[test]
    fn skip_gram_uses_gapped_context() {
        let mut store = Store::new(32);
        let (a, b, c, want, z) = (20u32, 21, 22, 23, 24);
        for _ in 0..50 {
            store.push_utterance(&[c, z]);
        }
        for _ in 0..8 {
            store.push_utterance(&[a, c, want]);
        }
        store.merge();
        let params = Params {
            f_min: 3,
            l_max: 8,
            max_gen_len: 1,
            tau_gen: 1.0,
            lambda_skip: 1.0,
            lambda_cache: 0.0,
            ..Params::default()
        };
        let smoothing = NaiveBackoff;
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut hit = 0;
        for _ in 0..40 {
            let g = generate_one(&store, &smoothing, &params, &[a, b, c], &[], &mut rng);
            if g.tokens.first() == Some(&want) {
                hit += 1;
            }
        }
        assert!(
            hit >= 36,
            "skip-gram [a,c] should dominate when [a,b,c] is missing; hit={hit}/40"
        );
    }

    #[test]
    fn cache_boosts_recent_history_when_context_unknown() {
        let mut store = Store::new(32);
        let (a, z) = (20u32, 24);
        for _ in 0..40 {
            store.push_utterance(&[z, z]);
        }
        store.merge();
        let params = Params {
            f_min: 3,
            l_max: 8,
            max_gen_len: 1,
            tau_gen: 1.0,
            lambda_skip: 0.0,
            lambda_cache: 1.0,
            ..Params::default()
        };
        let smoothing = NaiveBackoff;
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let mut hit = 0;
        for _ in 0..40 {
            let g = generate_one(&store, &smoothing, &params, &[a, a, a], &[], &mut rng);
            if g.tokens.first() == Some(&a) {
                hit += 1;
            }
        }
        assert!(
            hit >= 36,
            "recency cache should emit the in-context token when suffixes miss; hit={hit}/40"
        );
    }

    #[test]
    fn ppm_exclusion_drops_seen_types_from_backoff() {
        let back = vec![(1, 0.4), (2, 0.6)];
        let local = vec![(1, 5)];
        let out = exclude_seen(&back, &local);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, 2);
        assert!((out[0].1 - 1.0).abs() < 1e-9);
    }
}
