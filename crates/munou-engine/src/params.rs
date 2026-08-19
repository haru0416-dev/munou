use serde::{Deserialize, Serialize};

/// Tunable knobs from the design document §5.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Params {
    /// Markov candidate count `N_cand`.
    pub n_cand: usize,
    /// Generation temperature `τ_gen`.
    pub tau_gen: f64,
    /// Longest-match context cap `L_max` (chunks).
    pub l_max: usize,
    /// Backoff frequency threshold `f_min`.
    pub f_min: u32,
    /// Topic moving-average window `k_topic`.
    pub k_topic: usize,
    /// Trigger cosine threshold `θ_trig` — kept intentionally loose.
    pub theta_trig: f32,
    /// Slip injection rate `p_slip`.
    pub p_slip: f64,
    /// Hard cap on generated chunks.
    pub max_gen_len: usize,
    /// Generation-buffer size (tokens) that triggers a full SA rebuild.
    pub merge_threshold: usize,
    /// Character n-gram order for branching-entropy training.
    pub entropy_n: usize,
    /// Minimum entropy/AV score to cut inside a script run.
    pub entropy_cut: f64,
    /// Target morphemes per statistical chunk.
    pub chunk_morphs: usize,
    /// Embedding dimension for the closed hash embedder.
    pub embed_dim: usize,
    /// Similarity band `[a, b]` used by the eval metric (not the selector).
    pub band_lo: f32,
    pub band_hi: f32,
    /// Smoothing: `"naive"` or `"kn"`.
    pub smoothing: SmoothingKind,
    /// Absolute discount for Kneser-Ney.
    pub kn_discount: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SmoothingKind {
    Naive,
    Kn,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            n_cand: 10,
            tau_gen: 1.0,
            l_max: 8,
            f_min: 3,
            k_topic: 5,
            theta_trig: 0.42,
            p_slip: 0.15,
            max_gen_len: 24,
            merge_threshold: 4096,
            entropy_n: 5,
            entropy_cut: 0.8,
            chunk_morphs: 3,
            embed_dim: 256,
            band_lo: 0.25,
            band_hi: 0.85,
            smoothing: SmoothingKind::Naive,
            kn_discount: 0.75,
        }
    }
}

impl Params {
    pub fn l_max_capped(&self) -> usize {
        self.l_max.clamp(1, 8)
    }
}
