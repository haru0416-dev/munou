use crate::ids::TokenId;

/// Convert raw next-token counts into a sampling distribution.
/// Kneser-Ney is a drop-in replacement for the naive backoff.
pub trait Smoothing: Send + Sync {
    fn name(&self) -> &'static str;

    /// `local` is (token, count) in the matched context.
    /// `backoff` is the already-normalised lower-order distribution.
    /// `n1plus` is the number of unique continuations in this context.
    fn distribute(
        &self,
        local: &[(TokenId, u32)],
        total: u32,
        n1plus: u32,
        backoff: &[(TokenId, f64)],
    ) -> Vec<(TokenId, f64)>;
}

/// Maximum-likelihood with interpolation against the backoff distribution
/// when the context is sparse. The engine still shortens the context when
/// `total < f_min`; this layer only shapes the weights at a given order.
pub struct NaiveBackoff;

impl Smoothing for NaiveBackoff {
    fn name(&self) -> &'static str {
        "naive"
    }

    fn distribute(
        &self,
        local: &[(TokenId, u32)],
        total: u32,
        _n1plus: u32,
        backoff: &[(TokenId, f64)],
    ) -> Vec<(TokenId, f64)> {
        if total == 0 {
            return backoff.to_vec();
        }
        let lambda = total as f64 / (total as f64 + 1.0);
        mix(local, total, lambda, backoff)
    }
}

/// Interpolated absolute-discounting (Kneser-Ney shaped).
/// Unigram backoff should already be continuation-based when the store
/// provides it; this type does not itself compute left-context statistics.
pub struct KneserNey {
    pub discount: f64,
}

impl Smoothing for KneserNey {
    fn name(&self) -> &'static str {
        "kn"
    }

    fn distribute(
        &self,
        local: &[(TokenId, u32)],
        total: u32,
        n1plus: u32,
        backoff: &[(TokenId, f64)],
    ) -> Vec<(TokenId, f64)> {
        if total == 0 {
            return backoff.to_vec();
        }
        let d = self.discount.clamp(0.0, 1.0);
        let lambda = (d * n1plus as f64) / total as f64;
        let mut out: Vec<(TokenId, f64)> = Vec::new();
        let mut seen = rustc_hash::FxHashSet::default();
        for &(id, c) in local {
            let p = ((c as f64 - d).max(0.0) / total as f64) + lambda * lookup(backoff, id);
            out.push((id, p.max(0.0)));
            seen.insert(id);
        }
        for &(id, p_b) in backoff {
            if !seen.contains(&id) {
                out.push((id, lambda * p_b));
            }
        }
        let z: f64 = out.iter().map(|(_, p)| *p).sum();
        if z > 0.0 {
            for (_, p) in out.iter_mut() {
                *p /= z;
            }
        }
        out
    }
}

fn lookup(backoff: &[(TokenId, f64)], id: TokenId) -> f64 {
    backoff
        .iter()
        .find(|(t, _)| *t == id)
        .map(|(_, p)| *p)
        .unwrap_or(0.0)
}

fn mix(
    local: &[(TokenId, u32)],
    total: u32,
    lambda: f64,
    backoff: &[(TokenId, f64)],
) -> Vec<(TokenId, f64)> {
    let mut out: Vec<(TokenId, f64)> = Vec::new();
    let mut seen = rustc_hash::FxHashSet::default();
    for &(id, c) in local {
        let p = lambda * (c as f64 / total as f64) + (1.0 - lambda) * lookup(backoff, id);
        out.push((id, p));
        seen.insert(id);
    }
    for &(id, p_b) in backoff {
        if !seen.contains(&id) {
            out.push((id, (1.0 - lambda) * p_b));
        }
    }
    out
}

pub fn boxed(kind: crate::params::SmoothingKind, kn_discount: f64) -> Box<dyn Smoothing> {
    match kind {
        crate::params::SmoothingKind::Naive => Box::new(NaiveBackoff),
        crate::params::SmoothingKind::Kn => Box::new(KneserNey {
            discount: kn_discount,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naive_falls_back_when_empty() {
        let back = vec![(3, 1.0)];
        let out = NaiveBackoff.distribute(&[], 0, 0, &back);
        assert_eq!(out, back);
    }

    #[test]
    fn kn_discounts_and_renormalises() {
        let local = vec![(1, 10), (2, 1)];
        let back = vec![(1, 0.5), (2, 0.3), (3, 0.2)];
        let kn = KneserNey { discount: 0.75 };
        let out = kn.distribute(&local, 11, 2, &back);
        let z: f64 = out.iter().map(|(_, p)| *p).sum();
        assert!((z - 1.0).abs() < 1e-9);
        assert!(out.iter().any(|(id, _)| *id == 3));
    }
}
