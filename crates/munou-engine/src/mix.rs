//! Hybrid candidate pool: sources propose, the selector picks.
//!
//! v0.1 wired trigger XOR Markov, and Markov fell back to echoing the user.
//! The pool labels each proposal so parroting is a named source, not a silent
//! backoff, and retrieval / generation compete in one ranking.

use crate::explain::PathKind;
use crate::ids::TokenId;

#[derive(Debug, Clone)]
pub struct Proposal {
    pub source: PathKind,
    pub text: String,
    pub tokens: Vec<TokenId>,
    /// Mean −ln p per generated token (generated candidates only).
    pub surprise: Option<f32>,
}

#[derive(Debug, Default)]
pub struct Pool {
    pub items: Vec<Proposal>,
}

impl Pool {
    pub fn push(&mut self, source: PathKind, text: String, tokens: Vec<TokenId>) {
        self.push_scored(source, text, tokens, None);
    }

    pub fn push_scored(
        &mut self,
        source: PathKind,
        text: String,
        tokens: Vec<TokenId>,
        surprise: Option<f32>,
    ) {
        if text.trim().is_empty() {
            return;
        }
        if self.items.iter().any(|p| p.text == text) {
            return;
        }
        self.items.push(Proposal {
            source,
            text,
            tokens,
            surprise,
        });
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn texts(&self) -> Vec<String> {
        self.items.iter().map(|p| p.text.clone()).collect()
    }

    pub fn tokens(&self) -> Vec<Vec<TokenId>> {
        self.items.iter().map(|p| p.tokens.clone()).collect()
    }

    pub fn sources(&self) -> Vec<PathKind> {
        self.items.iter().map(|p| p.source).collect()
    }

    pub fn surprises(&self) -> Vec<Option<f32>> {
        self.items.iter().map(|p| p.surprise).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skips_duplicate_text() {
        let mut p = Pool::default();
        p.push(PathKind::Trigger, "おはよう".into(), vec![1]);
        p.push(PathKind::Echo, "おはよう".into(), vec![1]);
        p.push(PathKind::Markov, "散歩しよう".into(), vec![2]);
        assert_eq!(p.items.len(), 2);
        assert_eq!(p.items[0].source, PathKind::Trigger);
        assert_eq!(p.items[1].source, PathKind::Markov);
    }
}
