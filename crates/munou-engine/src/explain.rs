use serde::{Deserialize, Serialize};

use crate::ids::TokenId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PathKind {
    Trigger,
    Markov,
    Retrieve,
    Echo,
}

impl PathKind {
    pub fn tag(self) -> &'static str {
        match self {
            PathKind::Trigger => "trig",
            PathKind::Markov => "mark",
            PathKind::Retrieve => "retr",
            PathKind::Echo => "echo",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenStep {
    pub ctx_len_requested: usize,
    pub ctx_len_used: usize,
    pub freq: u32,
    pub sampled: TokenId,
    pub temperature: f32,
    /// Sampling probability after temperature + nucleus (decode analog of logprobs).
    pub p: f32,
    pub logp: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateTrace {
    pub rank: usize,
    pub source: PathKind,
    pub text: String,
    pub tokens: Vec<TokenId>,
    /// Topic cosine before source/rote adjustments.
    pub topic_score: f32,
    /// Ranking score (topic ± source prior − input-LCS penalty).
    pub score: f32,
    pub chosen: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerTrace {
    pub pattern: String,
    pub similarity: f32,
    pub threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    pub seed: u64,
    pub path: PathKind,
    pub input: String,
    pub morphemes: Vec<String>,
    pub chunks: Vec<String>,
    pub topic_hits: usize,
    pub trigger: Option<TriggerTrace>,
    pub candidates: Vec<CandidateTrace>,
    pub chosen_rank: usize,
    pub slipped: bool,
    pub slip_roll: f64,
    pub p_slip: f64,
    pub learned: bool,
    pub learn_roll: f64,
    pub p_learn: f64,
    pub steps: Vec<GenStep>,
    pub elapsed_us: u128,
    pub novelty_lcs: usize,
    pub similarity: f32,
    pub band_hit: bool,
    /// Closed analog of a tool/MoE trace. None on old in-memory tests that skip routing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route: Option<String>,
    /// RLHF-lite path prior from `/good` `/bad`.
    #[serde(default)]
    pub path_prior: [f32; 4],
}

impl Trace {
    pub fn explain_compact(&self) -> String {
        format!(
            "path={:?} learned={} sim={:.3} slipped={}",
            self.path, self.learned, self.similarity, self.slipped
        )
    }

    pub fn explain_text(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "path={:?}  elapsed={}us  slipped={} (p_slip={:.2} roll={:.3})\n",
            self.path, self.elapsed_us, self.slipped, self.p_slip, self.slip_roll
        ));
        s.push_str(&format!(
            "learned={} (p_learn={:.2} roll={:.3})\n",
            self.learned, self.p_learn, self.learn_roll
        ));
        if let Some(r) = &self.route {
            s.push_str(r);
            s.push('\n');
        }
        let pr = self.path_prior;
        if pr.iter().any(|x| *x != 0.0) {
            s.push_str(&format!(
                "pref trig={:+.2} mark={:+.2} retr={:+.2} echo={:+.2}\n",
                pr[0], pr[1], pr[2], pr[3]
            ));
        }
        s.push_str(&format!(
            "morph=[{}]  chunk=[{}]\n",
            self.morphemes.join(" / "),
            self.chunks.join(" / ")
        ));
        if let Some(t) = &self.trigger {
            s.push_str(&format!(
                "trigger \"{}\" sim={:.3} θ={:.3}\n",
                t.pattern, t.similarity, t.threshold
            ));
        }
        s.push_str(&format!(
            "sim={:.3} band_hit={} novelty_lcs={}\n",
            self.similarity, self.band_hit, self.novelty_lcs
        ));
        for c in &self.candidates {
            let mark = if c.chosen { ">" } else { " " };
            s.push_str(&format!(
                " {mark} #{:<2} {:+.3} [{}] topic={:+.3}  {}\n",
                c.rank + 1,
                c.score,
                c.source.tag(),
                c.topic_score,
                c.text
            ));
        }
        if !self.steps.is_empty() {
            s.push_str("gen:");
            for st in &self.steps {
                s.push_str(&format!(
                    "  ctx {}\u2192{} f={} tok={} p={:.3}",
                    st.ctx_len_requested, st.ctx_len_used, st.freq, st.sampled, st.p
                ));
            }
            s.push('\n');
        }
        s
    }
}
