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
mod route;
mod sais;
mod select;
mod smoothing;
mod store;
mod tokenizer;
mod trigger;

pub use engine::{Engine, OpenConfig, Reply, Stats};
pub use error::{Error, Result};
pub use eval::EvalAccum;
pub use explain::{CandidateTrace, GenStep, PathKind, Trace, TriggerTrace};
pub use generate::{lcs_len, lcsubstr_len};
pub use ids::{is_special, special_name, TokenId, BOS, EOS, FIRST_USER, SEP};
pub use intern::Interner;
pub use log::{AppendLog, Record, Role};
pub use observe::{Observe, Stage};
pub use params::{MixMode, Params, SmoothingKind};
pub use route::{plan as route_plan, RoutePlan};
pub use sais::{sa_range, suffix_array};
pub use tokenizer::{detokenize, Tokenizer};
pub use trigger::TriggerDict;

pub use embed::{cosine, Embedder, HashEmbedder, TopicTracker};
pub use smoothing::{KneserNey, NaiveBackoff, Smoothing};
pub use store::Store;
