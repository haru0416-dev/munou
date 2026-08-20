use serde::Deserialize;

use crate::embed::{cosine, Embedder};
use crate::explain::TriggerTrace;
use crate::params::Params;

#[derive(Debug, Clone, Deserialize)]
pub struct TriggerEntry {
    pub pattern: String,
    pub responses: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct TriggerDict {
    entries: Vec<TriggerEntry>,
}

impl TriggerDict {
    pub fn from_json(s: &str) -> crate::error::Result<Self> {
        let entries: Vec<TriggerEntry> = serde_json::from_str(s)?;
        Ok(Self { entries })
    }

    pub fn from_path(path: &std::path::Path) -> crate::error::Result<Self> {
        let raw = std::fs::read_to_string(path).map_err(|e| crate::error::Error::io(path, e))?;
        Self::from_json(&raw)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn match_one<E: Embedder>(
        &self,
        embedder: &E,
        input: &str,
        params: &Params,
    ) -> Option<(TriggerTrace, Vec<String>)> {
        if self.entries.is_empty() {
            return None;
        }
        let dim = embedder.dim();
        let mut q = vec![0.0f32; dim];
        let mut p = vec![0.0f32; dim];
        embedder.embed(input, &mut q);
        let mut best: Option<(f32, usize)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            embedder.embed(&e.pattern, &mut p);
            let sim = cosine(&q, &p);
            if best.map(|(s, _)| sim > s).unwrap_or(true) {
                best = Some((sim, i));
            }
        }
        let (sim, i) = best?;
        if sim < params.theta_trig {
            return None;
        }
        let e = &self.entries[i];
        if e.responses.is_empty() {
            return None;
        }
        Some((
            TriggerTrace {
                pattern: e.pattern.clone(),
                similarity: sim,
                threshold: params.theta_trig,
            },
            e.responses.clone(),
        ))
    }
}
