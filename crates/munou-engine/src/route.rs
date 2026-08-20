//! Closed analog of LLM *routing* (MoE / RAG vs weights).
//!
//! Experts are the existing PathKind sources. The gate is a function of
//! store occupancy and max retrieve cosine — no neural router, no API.

use crate::explain::PathKind;
use crate::params::Params;

#[derive(Debug, Clone, Copy)]
pub struct RoutePlan {
    pub g_trig: f32,
    pub g_retr: f32,
    pub g_mark: f32,
    pub g_echo: f32,
    pub n_retrieve: usize,
    pub n_cand: usize,
    pub run_retrieve: bool,
    pub run_markov: bool,
    pub run_echo: bool,
}

impl RoutePlan {
    pub fn explain_line(self) -> String {
        format!(
            "route g[trig={:.2} retr={:.2} mark={:.2} echo={:.2}]  n_retr={} n_cand={}",
            self.g_trig, self.g_retr, self.g_mark, self.g_echo, self.n_retrieve, self.n_cand
        )
    }
}

/// `retr_sim` = max topic cosine against past bot utterances (0 if none).
/// `trig_sim` = trigger pattern cosine, or 0 if no dict hit this turn.
pub fn plan(
    params: &Params,
    store_tokens: usize,
    n_bots: usize,
    retr_sim: f32,
    trig_sim: f32,
) -> RoutePlan {
    let retr_sim = retr_sim.clamp(0.0, 1.0);
    let g_trig = if params.theta_trig <= 1e-6 {
        0.0
    } else {
        (trig_sim / params.theta_trig).clamp(0.0, 1.0)
    };
    let g_retr = if n_bots == 0 { 0.0 } else { retr_sim };
    let g_mark = if store_tokens == 0 {
        0.0
    } else {
        (0.35 + 0.65 * (1.0 - 0.5 * g_retr)).clamp(0.2, 1.0)
    };
    // Echo stays available as the named fallback expert (pool tests rely on it).
    let g_echo = if store_tokens == 0 { 1.0 } else { 0.35 };

    let n_retrieve = if n_bots == 0 || params.n_retrieve == 0 {
        0
    } else {
        let s = 0.25 + 0.75 * g_retr;
        ((params.n_retrieve as f32 * s).round() as usize).clamp(1, params.n_retrieve)
    };
    let n_cand = if store_tokens == 0 || params.n_cand == 0 {
        0
    } else {
        ((params.n_cand as f32 * g_mark).round() as usize).max(1)
    };

    RoutePlan {
        g_trig,
        g_retr,
        g_mark,
        g_echo,
        n_retrieve,
        n_cand,
        run_retrieve: n_retrieve > 0 && params.mix == crate::params::MixMode::Pool,
        run_markov: n_cand > 0 || params.mix == crate::params::MixMode::Exclusive,
        run_echo: params.mix == crate::params::MixMode::Pool || store_tokens == 0,
    }
}

pub fn prior_index(p: PathKind) -> usize {
    match p {
        PathKind::Trigger => 0,
        PathKind::Markov => 1,
        PathKind::Retrieve => 2,
        PathKind::Echo => 3,
        PathKind::Adapt => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::Params;

    #[test]
    fn empty_store_skips_markov_capacity() {
        let p = Params::default();
        let r = plan(&p, 0, 0, 0.0, 0.0);
        assert_eq!(r.n_cand, 0);
        assert_eq!(r.n_retrieve, 0);
        assert!(r.run_echo);
    }

    #[test]
    fn high_retrieve_sim_allocates_retrieve_slots() {
        let p = Params::default();
        let cold = plan(&p, 100, 20, 0.05, 0.0);
        let hot = plan(&p, 100, 20, 0.9, 0.0);
        assert!(hot.n_retrieve >= cold.n_retrieve);
        assert!(hot.n_retrieve >= 1);
        assert!(hot.n_cand >= 1);
    }
}
