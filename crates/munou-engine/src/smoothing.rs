use crate::ids::TokenId;
use crate::params::{Params, SmoothingKind};
use crate::store::Store;

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

    /// Modified Kneser-Ney D1 / D2 / D3+. No-op for Witten-Bell.
    fn set_mkn(&mut self, d1: f64, d2: f64, d3: f64) {
        let _ = (d1, d2, d3);
    }

    fn mkn_discounts(&self) -> Option<(f64, f64, f64)> {
        None
    }
}

/// Maximum-likelihood with Witten-Bell interpolation against the backoff
/// distribution. `λ = N/(N+T)` where T is the number of unique continuations.
/// The engine interpolates every order; this layer only shapes one order.
pub struct NaiveBackoff;

impl Smoothing for NaiveBackoff {
    fn name(&self) -> &'static str {
        "naive"
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
        let n = total as f64;
        let t = n1plus.max(1) as f64;
        // Witten-Bell: λ = N/(N+T). More unique continuations → more backoff.
        let lambda = n / (n + t);
        mix(local, total, lambda, backoff)
    }
}

/// Modified Kneser-Ney (Chen & Goodman): count-dependent discounts D1/D2/D3+
/// and λ = (n1·D1 + n2·D2 + n3+·D3) / N. Unigram backoff should already be
/// continuation-based when the store provides it.
pub struct KneserNey {
    pub discount: f64,
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,
}

impl KneserNey {
    pub fn new(discount: f64) -> Self {
        let d = discount.max(0.0);
        Self {
            discount: d,
            d1: d.min(1.0),
            d2: d.min(2.0),
            d3: d.min(3.0),
        }
    }

    pub fn from_discounts(d1: f64, d2: f64, d3: f64) -> Self {
        Self {
            discount: d1,
            d1: d1.max(0.0),
            d2: d2.max(0.0),
            d3: d3.max(0.0),
        }
    }

    fn d_of(&self, c: u32) -> f64 {
        match c {
            0 => 0.0,
            1 => self.d1,
            2 => self.d2,
            _ => self.d3,
        }
        .min(c as f64)
    }
}

impl Smoothing for KneserNey {
    fn name(&self) -> &'static str {
        "kn"
    }

    fn set_mkn(&mut self, d1: f64, d2: f64, d3: f64) {
        self.d1 = d1.max(0.0);
        self.d2 = d2.max(0.0);
        self.d3 = d3.max(0.0);
    }

    fn mkn_discounts(&self) -> Option<(f64, f64, f64)> {
        Some((self.d1, self.d2, self.d3))
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
        let mut n1 = 0u32;
        let mut n2 = 0u32;
        let mut n3p = 0u32;
        for &(_, c) in local {
            match c {
                1 => n1 += 1,
                2 => n2 += 1,
                c if c >= 3 => n3p += 1,
                _ => {}
            }
        }
        let _ = n1plus;
        let lambda =
            (self.d1 * n1 as f64 + self.d2 * n2 as f64 + self.d3 * n3p as f64) / total as f64;
        let mut out: Vec<(TokenId, f64)> = Vec::with_capacity(local.len() + backoff.len());
        let bmap = backoff_values_for(local, backoff);
        for &(id, c) in local {
            let p_b = bmap.get(&id).copied().unwrap_or(0.0);
            let p = ((c as f64 - self.d_of(c)).max(0.0) / total as f64) + lambda * p_b;
            out.push((id, p.max(0.0)));
        }
        for &(id, p_b) in backoff {
            if !bmap.contains_key(&id) {
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

/// Backoff probabilities restricted to ids present in `local`, so the mix
/// avoids an O(|local|·|backoff|) linear search. Same values as before.
fn backoff_values_for(
    local: &[(TokenId, u32)],
    backoff: &[(TokenId, f64)],
) -> rustc_hash::FxHashMap<TokenId, f64> {
    let locals: rustc_hash::FxHashSet<TokenId> = local.iter().map(|(id, _)| *id).collect();
    let mut m = rustc_hash::FxHashMap::default();
    for &(id, p) in backoff {
        if locals.contains(&id) {
            m.insert(id, p);
        }
    }
    m
}

fn mix(
    local: &[(TokenId, u32)],
    total: u32,
    lambda: f64,
    backoff: &[(TokenId, f64)],
) -> Vec<(TokenId, f64)> {
    let mut out: Vec<(TokenId, f64)> = Vec::with_capacity(local.len() + backoff.len());
    let bmap = backoff_values_for(local, backoff);
    for &(id, c) in local {
        let p_b = bmap.get(&id).copied().unwrap_or(0.0);
        let p = lambda * (c as f64 / total as f64) + (1.0 - lambda) * p_b;
        out.push((id, p));
    }
    for &(id, p_b) in backoff {
        if !bmap.contains_key(&id) {
            out.push((id, (1.0 - lambda) * p_b));
        }
    }
    out
}

/// Chen & Goodman modified KN discounts from bigram count-of-counts.
/// Falls back to a single discount when n1 or n2 is missing.
pub fn chen_goodman(n1: u64, n2: u64, n3: u64, n4: u64, fallback: f64) -> (f64, f64, f64) {
    if n1 == 0 || n2 == 0 {
        let d = fallback.clamp(0.0, 0.95);
        return (d, d, d);
    }
    let y = n1 as f64 / (n1 as f64 + 2.0 * n2 as f64);
    let d1 = (1.0 - 2.0 * y * (n2 as f64 / n1 as f64)).clamp(0.05, 0.95);
    let d2 = (2.0 - 3.0 * y * (n3 as f64 / n2 as f64)).clamp(0.05, 1.95);
    let d3 = if n3 > 0 {
        (3.0 - 4.0 * y * (n4 as f64 / n3 as f64)).clamp(0.05, 2.95)
    } else {
        d2
    };
    (d1, d2, d3)
}

pub fn boxed(kind: SmoothingKind, kn_discount: f64) -> Box<dyn Smoothing> {
    match kind {
        SmoothingKind::Naive => Box::new(NaiveBackoff),
        SmoothingKind::Kn => Box::new(KneserNey::new(kn_discount)),
    }
}

/// Copy corpus MKN discounts onto the live smoother. `kn_discount == 0` keeps
/// discounts at zero (soak / ablation). This is not KenLM and does not load ARPA.
pub fn sync_to_store(sm: &mut dyn Smoothing, params: &Params, store: &Store) {
    if !matches!(params.smoothing, SmoothingKind::Kn) {
        return;
    }
    if params.kn_discount == 0.0 {
        sm.set_mkn(0.0, 0.0, 0.0);
        return;
    }
    let (d1, d2, d3) = store.mkn_discounts(params.kn_discount);
    sm.set_mkn(d1, d2, d3);
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
        let kn = KneserNey::new(0.75);
        let out = kn.distribute(&local, 11, 2, &back);
        let z: f64 = out.iter().map(|(_, p)| *p).sum();
        assert!((z - 1.0).abs() < 1e-9);
        assert!(out.iter().any(|(id, _)| *id == 3));
    }

    #[test]
    fn witten_bell_backs_off_more_when_types_grow() {
        let back = vec![(1, 0.5), (4, 0.5)];
        let few = NaiveBackoff.distribute(&[(1, 10)], 10, 1, &back);
        let many = NaiveBackoff.distribute(&[(1, 6), (2, 2), (3, 2)], 10, 3, &back);
        let p4 = |d: &[(TokenId, f64)]| {
            d.iter()
                .find(|(id, _)| *id == 4)
                .map(|(_, p)| *p)
                .unwrap_or(0.0)
        };
        assert!(
            p4(&many) > p4(&few),
            "T=3 should put more mass on unseen-in-context token than T=1; few={} many={}",
            p4(&few),
            p4(&many)
        );
    }

    #[test]
    fn chen_goodman_d1_le_d2_le_d3() {
        let (d1, d2, d3) = chen_goodman(100, 50, 20, 10, 0.75);
        assert!(d1 <= d2 + 1e-9, "d1={d1} d2={d2}");
        assert!(d2 <= d3 + 1e-9, "d2={d2} d3={d3}");
        assert!((0.05..=2.95).contains(&d1) && (0.05..=2.95).contains(&d3));
    }

    #[test]
    fn chen_goodman_falls_back_without_n1() {
        let (d1, d2, d3) = chen_goodman(0, 0, 0, 0, 0.75);
        assert!((d1 - 0.75).abs() < 1e-9);
        assert_eq!(d1, d2);
        assert_eq!(d2, d3);
    }

    #[test]
    fn mkn_lambda_uses_count_bins() {
        let local = vec![(1, 1), (2, 2), (3, 10)];
        let back = vec![(4, 1.0)];
        let kn = KneserNey::from_discounts(0.5, 1.0, 1.5);
        let out = kn.distribute(&local, 13, 3, &back);
        let p4 = out
            .iter()
            .find(|(id, _)| *id == 4)
            .map(|(_, p)| *p)
            .unwrap_or(0.0);
        assert!(p4 > 0.0, "backoff mass should reach unseen type");
        let z: f64 = out.iter().map(|(_, p)| *p).sum();
        assert!((z - 1.0).abs() < 1e-9);
    }
}
