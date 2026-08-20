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
    /// Probability of absorbing a live turn into the corpus (`p_learn`).
    /// The JSONL log always appends; this only gates tokenizer / SA / retrieve.
    pub p_learn: f64,
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
    /// Similarity band `[a, b]`. Eval metric and selector hinge.
    pub band_lo: f32,
    pub band_hi: f32,
    /// Hinge weight for sim outside the band. 0 restores pure cosine max.
    pub band_penalty: f32,
    /// Boltzmann temperature for p_slip sampling of ranks ≥ 2.
    pub tau_slip: f64,
    /// Smoothing: `"naive"` (Witten-Bell) or `"kn"` (modified Kneser-Ney).
    pub smoothing: SmoothingKind,
    /// Fallback absolute discount when bigram n1/n2 are too small. 0 disables MKN discounts.
    pub kn_discount: f64,
    /// Mix weight for a gapped (skip-gram) context. Applied only when the contiguous match is sparse.
    pub lambda_skip: f64,
    /// Mix weight for a recency unigram cache (Kuhn & De Mori analog). Sparse contexts only.
    pub lambda_cache: f64,
    /// PPM-C exclusion on Witten-Bell interpolation. Modified KN already excludes.
    pub ppm_exclude: bool,
    /// How candidate sources are combined.
    pub mix: MixMode,
    /// Max retrieved past bot utterances in the pool.
    pub n_retrieve: usize,
    /// How many recent bot utterances to scan for retrieve / route sim.
    /// 0 = scan all. Default keeps seed-scale logs unchanged and caps huge logs.
    pub n_retrieve_scan: usize,
    /// Echo proposals: 1 = exact user text, 2+ adds a mild shuffle.
    pub n_echo: usize,
    /// Subtracted from topic cosine when the source is Echo.
    pub echo_penalty: f32,
    /// Subtracted × (token longest-common-substring with the *input* / len).
    pub rote_penalty: f32,
    /// Subtracted × (longest run shared with a recent *own* reply / len).
    /// Breaks the retrieve→absorb→retrieve lock-in loop; the input-side
    /// rote penalty never saw the bot's own repetition.
    pub self_penalty: f32,
    /// How many recent bot replies the self-repetition penalty looks at.
    pub self_window: usize,
    /// Added to topic cosine when the source is Trigger.
    pub trigger_bonus: f32,
    /// Extra Trigger score × pattern-input cosine (strong hits beat retrieve).
    pub trigger_match_weight: f32,
    /// Subtracted when the source is Retrieve (slightly prefer recombination).
    pub retrieve_penalty: f32,
    /// Subtracted when the source is Adapt (modified copy sits between
    /// Markov recombination and Retrieve verbatim).
    pub adapt_penalty: f32,
    /// Max Adapt proposals: 1 = adapted reply, 2 = + a quoted past user line.
    pub n_adapt: usize,
    /// Keyword-anchored bidirectional generation (MegaHAL analog) on a
    /// reversed-stream twin store. Costs a second SA (memory and merge time
    /// roughly double on the store side).
    pub bidir: bool,
    /// Experimental selection term: score += weight × mean −ln p of the
    /// candidate's generation steps. 0 = record surprise in /why only.
    pub surprise_weight: f32,
    /// MMR λ for retrieve: λ·sim − (1−λ)·max redundancy. 1 = top-k cosine.
    pub mmr_lambda: f32,
    /// Nucleus mass `p`. 1 = keep the full distribution.
    pub p_nucleus: f64,
    /// Decode top-k. 0 = off.
    pub k_top: usize,
    /// `/good` `/bad` step on the last path prior.
    pub pref_step: f32,
    /// Clamp on each path prior.
    pub pref_clip: f32,
    /// 日和: deterministic per-day modulation of slip / 合いの手 / 口をつく,
    /// derived from (seed, day of the previous log record). Off = なぎ every day.
    #[serde(default = "d_true")]
    pub weather: bool,
    /// 合いの手 base probability per reply (weather-scaled). 0 disables.
    #[serde(default = "d_interject_rate")]
    pub interject_rate: f64,
    /// 関心 selection term: score += weight × max chunk interest. 0 disables.
    #[serde(default = "d_interest_weight")]
    pub interest_weight: f32,
    /// Bonus for candidates containing the day's 気になる語 (weather-scaled).
    #[serde(default = "d_care_bonus")]
    pub care_bonus: f32,
    /// 聞きかじり threshold: chunks in fewer distinct learned utterances carry
    /// no interest and are skipped as anchors.
    #[serde(default = "d_hearsay_min")]
    pub hearsay_min: u32,
    /// 口をつく: per-turn probability that hearsay chunks may anchor anyway.
    #[serde(default = "d_hearsay_release")]
    pub hearsay_release: f64,
}

fn d_true() -> bool {
    true
}
fn d_interject_rate() -> f64 {
    0.30
}
fn d_interest_weight() -> f32 {
    0.08
}
fn d_care_bonus() -> f32 {
    0.05
}
fn d_hearsay_min() -> u32 {
    2
}
fn d_hearsay_release() -> f64 {
    0.15
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SmoothingKind {
    Naive,
    Kn,
}

/// Candidate mixing. `pool` is the intended architecture; `exclusive` is v0.1 XOR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MixMode {
    /// Trigger hit replaces the pool; otherwise Markov (+ silent parrot fallback).
    Exclusive,
    /// Trigger, Markov, retrieve, and echo all propose; the selector picks.
    Pool,
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
            p_learn: 0.35,
            max_gen_len: 24,
            merge_threshold: 4096,
            entropy_n: 5,
            entropy_cut: 0.8,
            chunk_morphs: 3,
            embed_dim: 256,
            band_lo: 0.25,
            band_hi: 0.85,
            band_penalty: 0.5,
            tau_slip: 0.45,
            smoothing: SmoothingKind::Naive,
            kn_discount: 0.75,
            lambda_skip: 0.12,
            lambda_cache: 0.10,
            ppm_exclude: false,
            mix: MixMode::Pool,
            n_retrieve: 4,
            n_retrieve_scan: 1024,
            n_echo: 1,
            echo_penalty: 0.25,
            rote_penalty: 0.50,
            self_penalty: 0.60,
            self_window: 8,
            trigger_bonus: 0.10,
            trigger_match_weight: 1.0,
            retrieve_penalty: 0.20,
            adapt_penalty: 0.10,
            n_adapt: 2,
            bidir: true,
            surprise_weight: 0.0,
            mmr_lambda: 0.75,
            p_nucleus: 1.0,
            k_top: 0,
            pref_step: 0.08,
            pref_clip: 0.35,
            weather: true,
            interject_rate: 0.30,
            interest_weight: 0.08,
            care_bonus: 0.05,
            hearsay_min: 2,
            hearsay_release: 0.15,
        }
    }
}

impl Params {
    /// Effective context cap: `l_max` as given, minimum 1. Earlier versions
    /// silently clamped values above 8; the parameter is honoured now.
    pub fn l_max_capped(&self) -> usize {
        self.l_max.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l_max_above_eight_is_honoured() {
        let p = Params {
            l_max: 12,
            ..Params::default()
        };
        assert_eq!(p.l_max_capped(), 12);
        let zero = Params {
            l_max: 0,
            ..Params::default()
        };
        assert_eq!(zero.l_max_capped(), 1);
    }
}
