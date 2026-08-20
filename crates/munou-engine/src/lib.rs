//! `munou-engine` — LLM-free dialogue core for 人工無脳君.
//!
//! Generation (variable-order Markov over a suffix array) is separate from
//! selection (closed hash embeddings + slip). Trigger, retrieve, Markov, and
//! echo propose into one pool. Nothing here calls a network or a pretrained
//! generative model.
//!
//! Module map (boxes from design §3):
//!
//! | 設計の箱 | モジュール |
//! |---|---|
//! | トークナイザー（分岐エントロピー + AV） | `tokenizer`, `intern` |
//! | コーパスストア（u32 列 + SA-IS + unigram キャッシュ） | `store`, `sais` |
//! | マルコフ生成（dense 参照実装 / 疎表現の本番経路） | `generate`, `sparse`, `alias`, `smoothing` |
//! | トリガー / 検索 / エコー | `trigger`, `retrieve`（エコーは `engine` 内） |
//! | 候補プールと経路ゲート | `mix`, `route` |
//! | 選択器（話題余弦・帯域ヒンジ・slip） | `select`, `embed` |
//! | 記憶（append-only JSONL・吸収・再生） | `log`, `engine` |
//! | 観察窓・説明・評価 | `observe`, `explain`, `eval` |
//! | 大量ログの合成 | `fabricate` |

mod adapt;
mod alias;
mod embed;
mod engine;
mod error;
mod eval;
mod explain;
mod fabricate;
mod generate;
mod ids;
mod intern;
mod log;
mod mix;
mod observe;
mod params;
mod retrieve;
mod route;
mod sais;
mod select;
mod smoothing;
mod sparse;
mod store;
mod tokenizer;
mod trigger;

pub use engine::{Engine, OpenConfig, Reply, Stats};
pub use error::{Error, Result};
pub use eval::EvalAccum;
pub use explain::{CandidateTrace, GenStep, PathKind, Trace, TriggerTrace};
pub use fabricate::{records as fabricate_records, write_jsonl as fabricate_write, FabricateOpts};
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
pub use smoothing::{chen_goodman, KneserNey, NaiveBackoff, Smoothing};
pub use store::Store;
