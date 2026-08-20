//! Markov generation over interned chunks, with longest-match backoff.

use rand::Rng;

use crate::alias::{temper, AliasTable};
use crate::explain::GenStep;
use crate::ids::{TokenId, EOS};
use crate::params::Params;
use crate::smoothing::Smoothing;
use crate::store::Store;

type Sparse = (Vec<(TokenId, u32)>, u32, usize);

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
    let parrot_dist = parrot_unigram(parrot);

    let mut backoff = if unigram.is_empty() {
        parrot_dist.clone()
    } else {
        unigram
    };

    let mut used = 0usize;
    let mut freq = 0u32;
    let mut local: Vec<(TokenId, u32)> = Vec::new();
    let mut fallback: Option<Sparse> = None;

    for start in 0..=ctx.len() {
        let sub = &ctx[start..];
        if sub.is_empty() {
            continue;
        }
        let (counts, total) = store.next_counts(sub);
        if total > 0 && fallback.is_none() {
            fallback = Some((counts.clone(), total, sub.len()));
        }
        if total >= params.f_min {
            local = counts;
            freq = total;
            used = sub.len();
            fallback = None;
            break;
        }
    }
    if freq == 0 {
        if let Some((c, t, u)) = fallback {
            local = c;
            freq = t;
            used = u;
        }
    }

    // Walk from long to short but we iterated start 0..=len (longest first). Good.
    // If still empty, use unigram/parrot.
    let n1plus = local.len() as u32;
    let mixed = if freq > 0 {
        smoothing.distribute(&local, freq, n1plus, &backoff)
    } else {
        std::mem::take(&mut backoff)
    };

    if mixed.is_empty() {
        let pd = parrot_unigram(parrot);
        if pd.is_empty() {
            return (Vec::new(), Vec::new(), used, freq);
        }
        let ids: Vec<TokenId> = pd.iter().map(|(id, _)| *id).collect();
        let w: Vec<f64> = pd.iter().map(|(_, p)| *p).collect();
        return (ids, w, used, freq);
    }

    let ids: Vec<TokenId> = mixed.iter().map(|(id, _)| *id).collect();
    let w: Vec<f64> = mixed.iter().map(|(_, p)| *p).collect();
    (ids, w, used, freq)
}

fn parrot_unigram(parrot: &[TokenId]) -> Vec<(TokenId, f64)> {
    if parrot.is_empty() {
        return Vec::new();
    }
    let inv = 1.0 / parrot.len() as f64;
    // unique-preserving order
    let mut seen = rustc_hash::FxHashSet::default();
    let mut out = Vec::new();
    for &t in parrot {
        if seen.insert(t) {
            let c = parrot.iter().filter(|x| **x == t).count() as f64;
            out.push((t, c * inv));
        }
    }
    out
}

/// Longest common token subsequence length (novelty / rote-memorisation).
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
}
