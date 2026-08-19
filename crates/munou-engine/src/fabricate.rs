//! Closed conversation expander. Same life-log themes as `data/seed.jsonl`,
//! combined deterministically. This is not an external corpus.

use std::io::Write;
use std::path::Path;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::error::{Error, Result};
use crate::log::{Record, Role};

/// User/bot pairs in the seed log's register: walk, coffee, cat, work, game.
const PAIRS: &[(&str, &str)] = &[
    ("今日はいい天気だね", "うん、散歩しよう"),
    ("散歩しない？", "いいね、散歩しよう"),
    ("外歩きたい", "じゃあ散歩しよう"),
    ("公園いこ", "散歩しよう、公園"),
    ("また天気いいね", "散歩しよう"),
    ("コーヒー飲む？", "飲む、コーヒーがいい"),
    ("カフェいこ", "コーヒー飲もう"),
    ("眠いからコーヒー", "コーヒー飲む？"),
    ("朝はコーヒー", "私もコーヒー飲む"),
    ("もう一杯", "コーヒー飲もう"),
    ("猫かわいい", "うちの猫もかわいい"),
    ("猫なでたい", "猫かわいいね"),
    ("猫の動画みた", "猫かわいい"),
    ("猫飼ってる？", "うちの猫かわいい"),
    ("また猫", "猫かわいいね"),
    ("眠いね", "仕事つらい"),
    ("今日も仕事", "眠いね、仕事"),
    ("残業だ", "仕事つらいね"),
    ("休みたい", "仕事つらい、眠い"),
    ("まだ仕事？", "仕事つらい"),
    ("ゲームしよう", "今日はゲームしない"),
    ("何する？", "ゲームしようか"),
    ("暇だね", "ゲームしない？"),
    ("今夜空いてる？", "ゲームしよう"),
    ("またゲーム", "ゲームしようか"),
    ("おはよう", "おはよう"),
    ("ありがとう", "いえいえ"),
];

#[derive(Debug, Clone, Copy)]
pub struct FabricateOpts {
    /// Number of user/bot pairs (utterances = 2 × pairs).
    pub pairs: usize,
    pub rng_seed: u64,
    /// Fraction of turns that append a turn index so vocab can grow.
    /// 0 = recycle the seed register only (huge count, small vocab).
    pub unique_frac: f64,
}

impl Default for FabricateOpts {
    fn default() -> Self {
        Self {
            pairs: 1000,
            rng_seed: 1,
            unique_frac: 0.0,
        }
    }
}

/// Deterministic conversation records. Source of truth is still JSONL once written.
pub fn records(opts: FabricateOpts) -> Vec<Record> {
    let pairs = opts.pairs.max(1);
    let mut rng = ChaCha8Rng::seed_from_u64(opts.rng_seed);
    let unique_frac = opts.unique_frac.clamp(0.0, 1.0);
    let mut out = Vec::with_capacity(pairs * 2);
    let t0 = 1_700_000_000_000u64;
    let n_pref = pairs.min(PAIRS.len());
    for (i, &(u, b)) in PAIRS.iter().take(n_pref).enumerate() {
        push_pair(&mut out, &mut rng, unique_frac, t0, i, u, b);
    }
    for i in n_pref..pairs {
        let (u, b) = PAIRS[rng.gen_range(0..PAIRS.len())];
        push_pair(&mut out, &mut rng, unique_frac, t0, i, u, b);
    }
    out
}

fn push_pair(
    out: &mut Vec<Record>,
    rng: &mut ChaCha8Rng,
    unique_frac: f64,
    t0: u64,
    i: usize,
    u: &str,
    b: &str,
) {
    let (u, b) = if rng.gen::<f64>() < unique_frac {
        (format!("{u} {i}"), format!("{b} {i}"))
    } else {
        (u.to_string(), b.to_string())
    };
    let t = t0 + (i as u64) * 2;
    out.push(rec(t, Role::User, u));
    out.push(rec(t + 1, Role::Bot, b));
}

fn rec(t: u64, role: Role, text: String) -> Record {
    Record {
        v: 1,
        t,
        role,
        text,
        slipped: None,
        score: None,
        learned: true,
        path: None,
        novelty_lcs: None,
        n_tok: None,
    }
}

pub fn write_jsonl(path: &Path, opts: FabricateOpts) -> Result<usize> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
    }
    let recs = records(opts);
    let n = recs.len();
    let mut f = std::fs::File::create(path).map_err(|e| Error::io(path, e))?;
    for r in recs {
        let mut line = serde_json::to_string(&r)?;
        line.push('\n');
        f.write_all(line.as_bytes())
            .map_err(|e| Error::io(path, e))?;
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_even() {
        let a = records(FabricateOpts {
            pairs: 40,
            rng_seed: 3,
            unique_frac: 0.0,
        });
        let b = records(FabricateOpts {
            pairs: 40,
            rng_seed: 3,
            unique_frac: 0.0,
        });
        assert_eq!(a.len(), 80);
        assert_eq!(a[0].text, b[0].text);
        assert!(a.iter().any(|r| r.text == "おはよう"));
        assert_eq!(a.iter().filter(|r| r.role == Role::User).count(), 40);
    }
}
