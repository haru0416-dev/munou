//! Observation window: growth gauges from existing metrics.
//!
//! This is the product surface for 「育てられる」. It does not add a personality
//! model. Numbers come from the JSONL log, the suffix-array store, and `/eval`.

use crate::eval::EvalAccum;
use crate::explain::{PathKind, Trace};
use crate::log::{Record, Role};
use crate::params::Params;
use crate::Stats;
