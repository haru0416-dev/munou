use crate::explain::Trace;
use crate::log::{Record, Role};
use crate::params::Params;

#[derive(Debug, Clone, Default)]
pub struct EvalAccum {
    pub n: u32,
    pub band_hits: u32,
    pub slip_n: u32,
    pub sim_sum: f32,
    pub lcs_sum: usize,
    pub lcs_len_sum: usize,
}

impl EvalAccum {
    pub fn observe(&mut self, tr: &Trace, gen_len: usize) {
        self.n += 1;
        if tr.band_hit {
            self.band_hits += 1;
        }
        if tr.slipped {
            self.slip_n += 1;
        }
        self.sim_sum += tr.similarity;
        self.lcs_sum += tr.novelty_lcs;
        self.lcs_len_sum += gen_len;
    }

    /// Replay a bot JSONL line so `/eval` and gauges survive process restart.
    pub fn ingest_bot(&mut self, rec: &Record, params: &Params) {
        if rec.role != Role::Bot {
            return;
        }
        let Some(score) = rec.score else {
            return;
        };
        self.n += 1;
        if score >= params.band_lo && score <= params.band_hi {
            self.band_hits += 1;
        }
        if rec.slipped == Some(true) {
            self.slip_n += 1;
        }
        self.sim_sum += score;
        if let (Some(lcs), Some(n_tok)) = (rec.novelty_lcs, rec.n_tok) {
            self.lcs_sum += lcs;
            self.lcs_len_sum += n_tok;
        }
    }

    pub fn summary(&self, params: &Params) -> String {
        if self.n == 0 {
            return "eval: no turns yet".into();
        }
        let n = self.n as f32;
        format!(
            "eval n={}  band[{:.2},{:.2}]={:.1}%  mean_sim={:.3}  slip={:.1}%  rote_lcs={:.2}",
            self.n,
            params.band_lo,
            params.band_hi,
            100.0 * self.band_hits as f32 / n,
            self.sim_sum / n,
            100.0 * self.slip_n as f32 / n,
            if self.lcs_len_sum == 0 {
                0.0
            } else {
                self.lcs_sum as f32 / self.lcs_len_sum as f32
            }
        )
    }
}
