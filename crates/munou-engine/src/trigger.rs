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
    /// Pattern embeddings, filled once by `warm`. Same values the embedder
    /// produces; caching them stops re-embedding the whole dictionary every
    /// turn (the retrieve side got the same fix in v0.1.8).
    embs: Vec<Vec<f32>>,
}

impl TriggerDict {
    pub fn from_json(s: &str) -> crate::error::Result<Self> {
        let entries: Vec<TriggerEntry> = serde_json::from_str(s)?;
        Ok(Self {
            entries,
            embs: Vec::new(),
        })
    }

    /// Precompute pattern embeddings. Matching falls back to embedding on the
    /// fly when this has not been called, with identical results.
    pub fn warm<E: Embedder>(&mut self, embedder: &E) {
        let dim = embedder.dim();
        self.embs = self
            .entries
            .iter()
            .map(|e| {
                let mut v = vec![0.0f32; dim];
                embedder.embed(&e.pattern, &mut v);
                v
            })
            .collect();
    }

    pub fn from_path(path: &std::path::Path) -> crate::error::Result<Self> {
        let raw = std::fs::read_to_string(path).map_err(|e| crate::error::Error::io(path, e))?;
        Self::from_json(&raw)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `input_emb` is the caller's embedding of the input text (the engine
    /// already embeds it for the topic window; no second embed here).
    pub fn match_one<E: Embedder>(
        &self,
        embedder: &E,
        input_emb: &[f32],
        params: &Params,
    ) -> Option<(TriggerTrace, Vec<String>)> {
        if self.entries.is_empty() {
            return None;
        }
        let warmed = self.embs.len() == self.entries.len();
        let mut p = if warmed {
            Vec::new()
        } else {
            vec![0.0f32; embedder.dim()]
        };
        let mut best: Option<(f32, usize)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            let sim = if warmed {
                cosine(input_emb, &self.embs[i])
            } else {
                embedder.embed(&e.pattern, &mut p);
                cosine(input_emb, &p)
            };
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
