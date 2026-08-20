//! Sparse decode path: the interpolated distribution as explicit values for
//! touched ids plus one scalar on the base unigram for everything else.
//! `generate::dist_with_backoff` is the dense reference implementation; the
//! equivalence tests below pin the two together id by id.

use rand::Rng;

use crate::alias::AliasTable;
use crate::generate::{counts_to_ml, lookup_counts, parrot_unigram, NextMemo};
use crate::ids::TokenId;
use crate::params::Params;
use crate::smoothing::Smoothing;
use crate::store::{SamplingUnigram, Store};
use rustc_hash::FxHashMap;

/// Sparse mixture representation of the interpolated distribution:
/// explicit final values for touched ids plus one scalar on the base unigram
/// for every other id. Every operation in `dist_with_backoff` — Witten-Bell /
/// modified-KN interpolation, PPM exclusion, skip-gram and recency-cache
/// mixing — is affine in the base outside the touched support, so the
/// |V|-sized vector never needs to be materialised. Values equal the dense
/// path up to floating-point association.
pub(crate) struct SparseDist {
    ids: Vec<TokenId>,
    vals: Vec<f64>,
    idx: FxHashMap<TokenId, usize>,
    /// P(w) = tail · U(w) for every w not in `ids`.
    tail: f64,
    /// Σ U(id) over explicit ids (base mass the tail no longer covers).
    umass: f64,
}

impl SparseDist {
    fn new() -> Self {
        Self {
            ids: Vec::new(),
            vals: Vec::new(),
            idx: FxHashMap::default(),
            tail: 1.0,
            umass: 0.0,
        }
    }

    fn u(uni: &SamplingUnigram<'_>, id: TokenId) -> f64 {
        uni.map.get(&id).copied().unwrap_or(0.0)
    }

    fn current(&self, uni: &SamplingUnigram<'_>, id: TokenId) -> f64 {
        match self.idx.get(&id) {
            Some(&i) => self.vals[i],
            None => self.tail * Self::u(uni, id),
        }
    }

    fn promote(&mut self, uni: &SamplingUnigram<'_>, id: TokenId) -> usize {
        if let Some(&i) = self.idx.get(&id) {
            return i;
        }
        let uv = Self::u(uni, id);
        let i = self.ids.len();
        self.ids.push(id);
        self.vals.push(self.tail * uv);
        self.idx.insert(id, i);
        self.umass += uv;
        i
    }

    fn scale(&mut self, k: f64) {
        for v in &mut self.vals {
            *v *= k;
        }
        self.tail *= k;
    }

    fn total(&self, u_total: f64) -> f64 {
        let s: f64 = self.vals.iter().map(|v| v.max(0.0)).sum();
        s + (self.tail * (u_total - self.umass)).max(0.0)
    }

    /// PPM-C exclusion: zero the ids seen at this order, renormalise the rest.
    fn exclude(&mut self, counts: &[(TokenId, u32)], uni: &SamplingUnigram<'_>, u_total: f64) {
        for &(id, _) in counts {
            let i = self.promote(uni, id);
            self.vals[i] = 0.0;
        }
        let z = self.total(u_total);
        if z > 0.0 {
            self.scale(1.0 / z);
        }
    }

    /// Witten-Bell: P_new = λ·ML + (1−λ)·P_old with λ = N/(N+T).
    fn wb_order(&mut self, counts: &[(TokenId, u32)], total: u32, uni: &SamplingUnigram<'_>) {
        let n = total as f64;
        let t = (counts.len() as u32).max(1) as f64;
        let lambda = n / (n + t);
        let olds: Vec<f64> = counts
            .iter()
            .map(|&(id, _)| self.current(uni, id))
            .collect();
        self.scale(1.0 - lambda);
        for (&(id, c), p_old) in counts.iter().zip(olds) {
            let i = self.promote(uni, id);
            self.vals[i] = lambda * (c as f64 / n) + (1.0 - lambda) * p_old;
        }
    }

    /// Modified KN: P_new = (c−D(c))⁺/N + λ·P_old, then renormalise.
    #[allow(clippy::too_many_arguments)]
    fn kn_order(
        &mut self,
        counts: &[(TokenId, u32)],
        total: u32,
        d1: f64,
        d2: f64,
        d3: f64,
        uni: &SamplingUnigram<'_>,
        u_total: f64,
    ) {
        let n = total as f64;
        let mut n1 = 0u32;
        let mut n2 = 0u32;
        let mut n3p = 0u32;
        for &(_, c) in counts {
            match c {
                1 => n1 += 1,
                2 => n2 += 1,
                c if c >= 3 => n3p += 1,
                _ => {}
            }
        }
        let lambda = (d1 * n1 as f64 + d2 * n2 as f64 + d3 * n3p as f64) / n;
        let d_of = |c: u32| -> f64 {
            match c {
                0 => 0.0,
                1 => d1,
                2 => d2,
                _ => d3,
            }
            .min(c as f64)
        };
        let olds: Vec<f64> = counts
            .iter()
            .map(|&(id, _)| self.current(uni, id))
            .collect();
        self.scale(lambda);
        for (&(id, c), p_old) in counts.iter().zip(olds) {
            let i = self.promote(uni, id);
            self.vals[i] = (((c as f64 - d_of(c)).max(0.0) / n) + lambda * p_old).max(0.0);
        }
        let z = self.total(u_total);
        if z > 0.0 {
            self.scale(1.0 / z);
        }
    }

    /// `P_new = (1−λ)·P_old + λ·extra`, then renormalise (mirrors `mix_lambda`).
    fn mix(&mut self, extra: &[(TokenId, f64)], lam: f64, uni: &SamplingUnigram<'_>, u_total: f64) {
        let lam = lam.clamp(0.0, 1.0);
        if lam <= 0.0 || extra.is_empty() {
            return;
        }
        let keep = 1.0 - lam;
        if keep <= 0.0 {
            self.ids.clear();
            self.vals.clear();
            self.idx.clear();
            self.tail = 0.0;
            self.umass = 0.0;
            for &(id, p) in extra {
                let i = self.ids.len();
                self.ids.push(id);
                self.vals.push(p);
                self.idx.insert(id, i);
                self.umass += Self::u(uni, id);
            }
        } else {
            let olds: Vec<f64> = extra.iter().map(|&(id, _)| self.current(uni, id)).collect();
            self.scale(keep);
            for (&(id, p_e), p_old) in extra.iter().zip(olds) {
                let i = self.promote(uni, id);
                self.vals[i] = keep * p_old + lam * p_e;
            }
        }
        let z = self.total(u_total);
        if z > 0.0 {
            self.scale(1.0 / z);
        }
    }

    /// Two-level draw: alias over the explicit support, or the cached unigram
    /// alias for the tail with rejection of promoted ids. Returns the token
    /// and its normalised probability.
    fn sample<R: Rng + ?Sized>(
        &self,
        uni: &SamplingUnigram<'_>,
        u_total: f64,
        rng: &mut R,
    ) -> Option<(TokenId, f32)> {
        let m_s: f64 = self.vals.iter().map(|v| v.max(0.0)).sum();
        let m_t = (self.tail * (u_total - self.umass)).max(0.0);
        let z = m_s + m_t;
        if z <= 0.0 {
            return None;
        }
        let r: f64 = rng.gen::<f64>() * z;
        if r < m_s || m_t <= 0.0 {
            let table = AliasTable::from_weights(&self.vals);
            let i = table.sample(rng);
            let p = (self.vals[i].max(0.0) / z) as f32;
            Some((self.ids[i], p))
        } else {
            for _ in 0..64 {
                let i = uni.alias.sample(rng);
                let id = uni.dist[i].0;
                if !self.idx.contains_key(&id) {
                    let p = (self.tail * uni.dist[i].1 / z) as f32;
                    return Some((id, p));
                }
            }
            // Explicit support covers most of the base mass — materialise the
            // complement once (rare).
            let mut cids: Vec<TokenId> = Vec::new();
            let mut cw: Vec<f64> = Vec::new();
            for &(id, pu) in uni.dist {
                if !self.idx.contains_key(&id) {
                    cids.push(id);
                    cw.push(pu);
                }
            }
            if cids.is_empty() {
                return None;
            }
            let table = AliasTable::from_weights(&cw);
            let i = table.sample(rng);
            Some((cids[i], (self.tail * cw[i] / z) as f32))
        }
    }
}

/// One decode step on the sparse representation. Same interpolation maths as
/// `dist_with_backoff`; returns (token, p, used_len, freq) or None to stop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sparse_step<R: Rng + ?Sized>(
    store: &Store,
    smoothing: &dyn Smoothing,
    params: &Params,
    ctx: &[TokenId],
    parrot: &[TokenId],
    uni: SamplingUnigram<'_>,
    u_total: f64,
    memo: &mut NextMemo,
    rng: &mut R,
) -> Option<(TokenId, f32, usize, u32)> {
    let (dist, used, freq) = sparse_dist(store, smoothing, params, ctx, uni, u_total, memo);
    match dist.sample(&uni, u_total, rng) {
        Some((id, p)) => Some((id, p, used, freq)),
        None => {
            // Mirror the dense path: an empty distribution falls back to the
            // parrot unigram before giving up.
            let pu = parrot_unigram(parrot);
            if pu.is_empty() {
                return None;
            }
            let w: Vec<f64> = pu.iter().map(|(_, p)| *p).collect();
            let z: f64 = w.iter().copied().filter(|x| *x > 0.0).sum();
            let table = AliasTable::from_weights(&w);
            let i = table.sample(rng);
            let p = if z > 0.0 {
                (w[i].max(0.0) / z) as f32
            } else {
                0.0
            };
            Some((pu[i].0, p, used, freq))
        }
    }
}

/// Build the sparse interpolated distribution for one context.
pub(crate) fn sparse_dist(
    store: &Store,
    smoothing: &dyn Smoothing,
    params: &Params,
    ctx: &[TokenId],
    uni: SamplingUnigram<'_>,
    u_total: f64,
    memo: &mut NextMemo,
) -> (SparseDist, usize, u32) {
    let mut dist = SparseDist::new();
    let mut used = 0usize;
    let mut freq = 0u32;
    let exclude =
        params.ppm_exclude || matches!(params.smoothing, crate::params::SmoothingKind::Kn);
    let kn_ds = smoothing.mkn_discounts();
    for len in 1..=ctx.len() {
        let sub = &ctx[ctx.len() - len..];
        let (counts, total) = lookup_counts(store, memo, sub);
        if total == 0 {
            continue;
        }
        if exclude {
            dist.exclude(&counts, &uni, u_total);
        }
        match kn_ds {
            Some((d1, d2, d3)) => dist.kn_order(&counts, total, d1, d2, d3, &uni, u_total),
            None => dist.wb_order(&counts, total, &uni),
        }
        if total >= params.f_min || freq < params.f_min {
            used = len;
            freq = total;
        }
    }

    let sparse_ctx = used < ctx.len() || freq < params.f_min;
    if sparse_ctx && params.lambda_skip > 0.0 && ctx.len() >= 3 {
        let mut skip = ctx.to_vec();
        skip.remove(skip.len() - 2);
        let (counts, total) = lookup_counts(store, memo, &skip);
        if total > 0 {
            let extra = counts_to_ml(&counts, total);
            dist.mix(&extra, params.lambda_skip, &uni, u_total);
        }
    }
    if sparse_ctx && params.lambda_cache > 0.0 && !ctx.is_empty() {
        let hist = if ctx.len() > 12 {
            &ctx[ctx.len() - 12..]
        } else {
            ctx
        };
        let extra = parrot_unigram(hist);
        dist.mix(&extra, params.lambda_cache, &uni, u_total);
    }

    (dist, used, freq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::dist_with_backoff;
    use crate::smoothing::NaiveBackoff;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// The sparse representation must equal the dense reference for every
    /// vocabulary id, for every smoothing / exclusion / mixing combination.
    /// Deterministic value comparison — no sampling statistics involved.
    #[test]
    fn sparse_dist_matches_dense_reference() {
        use crate::params::SmoothingKind;
        use crate::smoothing;

        // Deterministic pseudo-random corpus: vocab 20..=119, varied lengths.
        let mut store = Store::new(64);
        let mut x = 0x9e3779b97f4a7c15u64;
        let mut rnd = move || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 33) as u32
        };
        for _ in 0..400 {
            let len = 2 + (rnd() % 6) as usize;
            let utt: Vec<TokenId> = (0..len).map(|_| 20 + rnd() % 100).collect();
            store.push_utterance(&utt);
        }
        // leave some utterances in the buffer to exercise the linear path
        for _ in 0..5 {
            let utt: Vec<TokenId> = (0..3).map(|_| 20 + rnd() % 100).collect();
            store.push_utterance(&utt);
        }

        let contexts: Vec<Vec<TokenId>> = vec![
            vec![20 + rnd() % 100],
            vec![20 + rnd() % 100, 20 + rnd() % 100],
            vec![777, 888, 20 + rnd() % 100], // mostly unseen → skip/cache fire
            vec![20 + rnd() % 100, 999, 20 + rnd() % 100, 20 + rnd() % 100],
        ];

        for (kind, ppm) in [
            (SmoothingKind::Naive, false),
            (SmoothingKind::Naive, true),
            (SmoothingKind::Kn, false),
        ] {
            let params = Params {
                smoothing: kind,
                ppm_exclude: ppm,
                f_min: 3,
                l_max: 8,
                lambda_skip: 0.12,
                lambda_cache: 0.10,
                ..Params::default()
            };
            let mut sm = smoothing::boxed(kind, params.kn_discount);
            smoothing::sync_to_store(sm.as_mut(), &params, &store);
            let kn = matches!(kind, SmoothingKind::Kn);
            store.warm_sampling(kn);
            let uni = store.sampling_view(kn).expect("warmed");
            let u_total: f64 = uni.dist.iter().map(|(_, p)| *p).sum();

            for ctx in &contexts {
                let mut memo_d = NextMemo::default();
                let (ids, w, used_d, freq_d) = dist_with_backoff(
                    &store,
                    sm.as_ref(),
                    &params,
                    ctx,
                    &[],
                    uni.dist,
                    &mut memo_d,
                );
                let z_d: f64 = w.iter().copied().filter(|x| *x > 0.0).sum();
                assert!(z_d > 0.0, "dense empty for ctx={ctx:?}");

                let mut memo_s = NextMemo::default();
                let (sd, used_s, freq_s) =
                    sparse_dist(&store, sm.as_ref(), &params, ctx, uni, u_total, &mut memo_s);
                let z_s = sd.total(u_total);
                assert!(z_s > 0.0);
                assert_eq!((used_d, freq_d), (used_s, freq_s), "ctx={ctx:?}");

                let mut sum_abs_diff = 0.0f64;
                for (id, wv) in ids.iter().zip(&w) {
                    let p_d = wv.max(0.0) / z_d;
                    let p_s = sd.current(&uni, *id).max(0.0) / z_s;
                    let tol = 1e-9 + 1e-6 * p_d;
                    assert!(
                        (p_d - p_s).abs() <= tol,
                        "kind={kind:?} ppm={ppm} ctx={ctx:?} id={id} dense={p_d} sparse={p_s}"
                    );
                    sum_abs_diff += (p_d - p_s).abs();
                }
                assert!(sum_abs_diff < 1e-6, "L1 drift {sum_abs_diff}");
            }
        }
    }

    /// Sampling from the sparse representation matches its own distribution
    /// (two-level draw incl. tail rejection), checked empirically.
    #[test]
    fn sparse_sample_follows_distribution() {
        let mut store = Store::new(64);
        for _ in 0..30 {
            store.push_utterance(&[20, 21]);
        }
        for _ in 0..10 {
            store.push_utterance(&[20, 22]);
        }
        for i in 0..40u32 {
            store.push_utterance(&[30 + i, 30 + ((i + 1) % 40)]);
        }
        store.merge();
        let params = Params {
            f_min: 3,
            l_max: 8,
            ..Params::default()
        };
        let sm = NaiveBackoff;
        store.warm_sampling(false);
        let uni = store.sampling_view(false).expect("warmed");
        let u_total: f64 = uni.dist.iter().map(|(_, p)| *p).sum();
        let ctx = vec![20u32];
        let mut memo = NextMemo::default();
        let (sd, _, _) = sparse_dist(&store, &sm, &params, &ctx, uni, u_total, &mut memo);
        let z = sd.total(u_total);
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let n = 60_000usize;
        let mut hist: FxHashMap<TokenId, u32> = FxHashMap::default();
        for _ in 0..n {
            let (id, _) = sd.sample(&uni, u_total, &mut rng).expect("non-empty");
            *hist.entry(id).or_insert(0) += 1;
        }
        // every id with mass ≥ 1% must appear at its probability ± 5σ
        for &(id, _) in uni.dist {
            let p = sd.current(&uni, id).max(0.0) / z;
            if p < 0.01 {
                continue;
            }
            let got = hist.get(&id).copied().unwrap_or(0) as f64 / n as f64;
            let sigma = (p * (1.0 - p) / n as f64).sqrt();
            assert!(
                (got - p).abs() <= 5.0 * sigma + 1e-4,
                "id={id} p={p} got={got}"
            );
        }
    }
}
