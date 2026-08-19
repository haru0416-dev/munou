use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rustc_hash::FxHashSet;

use crate::embed::{cosine, Embedder, HashEmbedder, TopicTracker};
use crate::error::Result;
use crate::eval::EvalAccum;
use crate::explain::{PathKind, Trace};
use crate::generate::{generate_one, lcsubstr_len};
use crate::ids::{is_special, TokenId};
use crate::intern::Interner;
use crate::log::{now_ms, AppendLog, Record, Role};
use crate::mix::Pool;
use crate::observe::Observe;
use crate::params::{MixMode, Params};
use crate::route;
use crate::select::{rank_and_pick, RankInput};
use crate::smoothing::{self, Smoothing};
use crate::store::Store;
use crate::tokenizer::{detokenize, Tokenized, Tokenizer};
use crate::trigger::TriggerDict;

#[derive(Default)]
pub struct OpenConfig {
    pub params: Params,
    pub seed: u64,
    pub log_path: Option<PathBuf>,
    pub triggers_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct Reply {
    pub text: String,
    pub trace: Trace,
}

#[derive(Debug, Clone)]
pub struct Stats {
    pub utterances: usize,
    pub learned: usize,
    pub tokens: usize,
    pub vocab: usize,
    pub buf: usize,
    pub topic_window: usize,
    pub episodic: usize,
    pub meta: usize,
    pub hist: usize,
    pub path_prior: [f32; 4],
}

pub struct Engine {
    intern: Interner,
    tokenizer: Tokenizer,
    store: Store,
    embedder: HashEmbedder,
    topic: TopicTracker,
    triggers: TriggerDict,
    params: Params,
    rng: ChaCha8Rng,
    seed: u64,
    log: AppendLog,
    last_trace: Option<Trace>,
    smoothing: Box<dyn Smoothing>,
    eval: EvalAccum,
    history: VecDeque<TokenId>,
    prior: Vec<Vec<TokenId>>,
    /// Past bot utterances for retrieval (text + chunk ids).
    bots: Vec<(String, Vec<TokenId>)>,
    /// Closed analog of RLHF: additive path prior from `/good` `/bad`.
    path_prior: [f32; 4],
    /// Last bot path, restored from the log so `/good` works after reopen.
    last_path: Option<PathKind>,
}
