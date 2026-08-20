//! Vose's alias method: O(n) preprocess, O(1) sample.

use rand_core::Rng;

#[derive(Debug, Clone)]
pub struct AliasTable {
    n: usize,
    prob: Vec<f64>,
    alias: Vec<usize>,
}

impl AliasTable {
    /// Build from non-negative weights. Degenerate (all-zero / empty) tables
    /// sample index 0 of a 1-element dummy so callers can still proceed.
    pub fn from_weights(weights: &[f64]) -> Self {
        let n = weights.len();
        if n == 0 {
            return Self {
                n: 1,
                prob: vec![1.0],
                alias: vec![0],
            };
        }
        let sum: f64 = weights.iter().copied().filter(|w| *w > 0.0).sum();
        if sum <= 0.0 {
            return Self {
                n,
                prob: vec![1.0; n],
                alias: (0..n).collect(),
            };
        }
        let mut scaled: Vec<f64> = weights
            .iter()
            .map(|&w| {
                let w = if w > 0.0 { w } else { 0.0 };
                w * (n as f64) / sum
            })
            .collect();

        let mut small = Vec::new();
        let mut large = Vec::new();
        for (i, &p) in scaled.iter().enumerate() {
            if p < 1.0 {
                small.push(i);
            } else {
                large.push(i);
            }
        }

        let mut prob = vec![0.0; n];
        let mut alias = vec![0; n];

        while !small.is_empty() && !large.is_empty() {
            let s = small.pop().unwrap();
            let l = large.pop().unwrap();
            prob[s] = scaled[s];
            alias[s] = l;
            scaled[l] = (scaled[l] + scaled[s]) - 1.0;
            if scaled[l] < 1.0 {
                small.push(l);
            } else {
                large.push(l);
            }
        }
        while let Some(l) = large.pop() {
            prob[l] = 1.0;
            alias[l] = l;
        }
        while let Some(s) = small.pop() {
            prob[s] = 1.0;
            alias[s] = s;
        }

        Self { n, prob, alias }
    }

    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let i = crate::rng::rand_below(rng, self.n);
        let coin: f64 = crate::rng::rand_f64(rng);
        if coin < self.prob[i] {
            i
        } else {
            self.alias[i]
        }
    }

    /// Boltzmann weights: `exp((s_i - max)/τ)`. Numerically stable.
    pub fn softmax_weights(scores: &[f64], tau: f64) -> Vec<f64> {
        if scores.is_empty() {
            return Vec::new();
        }
        let tau = tau.max(1e-6);
        let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !m.is_finite() {
            return vec![1.0; scores.len()];
        }
        scores.iter().map(|s| ((s - m) / tau).exp()).collect()
    }

    pub fn softmax_sample<R: Rng + ?Sized>(scores: &[f64], tau: f64, rng: &mut R) -> usize {
        let w = Self::softmax_weights(scores, tau);
        Self::from_weights(&w).sample(rng)
    }
}

/// Nucleus (top-p) then optional top-k. Always keeps ≥1 mass-bearing entry.
/// `p >= 1` and `k == 0` is a no-op. This is the closed analog of LLM decode
/// filters — the distribution still comes from the suffix array.
pub fn nucleus(ids: &mut Vec<crate::ids::TokenId>, weights: &mut Vec<f64>, p: f64, k: usize) {
    let n = weights.len().min(ids.len());
    if n == 0 {
        return;
    }
    let p = p.clamp(0.0, 1.0);
    if p >= 1.0 - 1e-12 && k == 0 {
        return;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        weights[b]
            .partial_cmp(&weights[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if k > 0 && k < order.len() {
        order.truncate(k);
    }
    let tot: f64 = order.iter().map(|&i| weights[i].max(0.0)).sum();
    if tot <= 0.0 {
        return;
    }
    let thresh = p * tot;
    let mut acc = 0.0;
    let mut keep = vec![false; n];
    for i in order {
        keep[i] = true;
        acc += weights[i].max(0.0);
        if acc + 1e-12 >= thresh {
            break;
        }
    }
    let mut ni = Vec::new();
    let mut nw = Vec::new();
    for i in 0..n {
        if keep[i] {
            ni.push(ids[i]);
            nw.push(weights[i]);
        }
    }
    if ni.is_empty() {
        return;
    }
    *ids = ni;
    *weights = nw;
}

/// Apply temperature `τ`: `p_i^{1/τ}` then renormalise. `τ == 1` is identity.
pub fn temper(weights: &mut [f64], tau: f64) {
    if (tau - 1.0).abs() < 1e-12 {
        return;
    }
    let inv = 1.0 / tau.max(1e-6);
    for w in weights.iter_mut() {
        if *w > 0.0 {
            *w = w.powf(inv);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    #[test]
    fn alias_matches_weights_roughly() {
        let w = [1.0, 2.0, 3.0, 4.0];
        let table = AliasTable::from_weights(&w);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut hist = [0u32; 4];
        let n = 40_000u32;
        for _ in 0..n {
            hist[table.sample(&mut rng)] += 1;
        }
        let total: f64 = w.iter().sum();
        for i in 0..4 {
            let got = hist[i] as f64 / n as f64;
            let exp = w[i] / total;
            assert!((got - exp).abs() < 0.02, "i={i} got={got} exp={exp}");
        }
    }

    #[test]
    fn temper_identity() {
        let mut w = [0.2, 0.8];
        temper(&mut w, 1.0);
        assert!((w[0] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn softmax_peaks_on_max() {
        let w = AliasTable::softmax_weights(&[0.0, 2.0, 0.0], 0.5);
        assert!(w[1] > w[0] && w[1] > w[2]);
        let z: f64 = w.iter().sum();
        assert!(z > 0.0);
    }

    #[test]
    fn nucleus_keeps_head_mass() {
        let mut ids = vec![1u32, 2, 3, 4];
        let mut w = vec![0.7, 0.2, 0.05, 0.05];
        nucleus(&mut ids, &mut w, 0.9, 0);
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert_eq!(ids.len(), 2);
    }
}
