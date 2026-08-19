//! `munou-engine` — LLM-free dialogue core for 人工無脳君.
//!
//! Generation (variable-order Markov over a suffix array) is separate from
//! selection (closed hash embeddings + slip). Trigger, retrieve, Markov, and
//! echo propose into one pool. Nothing here calls a network or a pretrained
//! generative model.

mod alias;
mod embed;
mod engine;
mod error;
mod eval;
mod explain;
mod generate;
mod ids;
mod intern;
mod log;
mod mix;
mod observe;
mod params;
mod sais;
mod select;
mod smoothing;
mod store;
mod tokenizer;
mod trigger;

pub use engine::{Engine, OpenConfig, Reply, Stats};
