//! Closed conversation expander. Same life-log themes as `data/seed.jsonl`,
//! combined deterministically. This is not an external corpus.

use std::io::Write;
use std::path::Path;

use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;

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

/// Syllabary for coined words. Closed and synthetic; kana rather than digit
/// suffixes because fabricated surface leaks into replies (「ゲームしよう
/// 4999」 otherwise).
const MORAE: &[&str] = &[
    "か", "き", "く", "け", "こ", "さ", "し", "す", "そ", "た", "ち", "つ", "て", "と", "な", "に",
    "ぬ", "ね", "の", "は", "ひ", "ふ", "へ", "ほ", "ま", "み", "む", "め", "も", "や", "ゆ", "よ",
    "ら", "り", "る", "れ", "ろ", "わ",
];

fn coin_word(rng: &mut ChaCha8Rng) -> String {
    let n = 2 + crate::rng::rand_below(rng, 3); // 2..=4 morae
    (0..n)
        .map(|_| MORAE[crate::rng::rand_below(rng, MORAE.len())])
        .collect()
}

#[derive(Debug, Clone, Copy)]
pub struct FabricateOpts {
    /// Number of user/bot pairs (utterances = 2 × pairs).
    pub pairs: usize,
    pub rng_seed: u64,
    /// Fraction of turns that append a coined kana word so vocab can grow.
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
        let (u, b) = PAIRS[crate::rng::rand_below(&mut rng, PAIRS.len())];
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
    let (u, b) = if crate::rng::rand_f64(rng) < unique_frac {
        let w = coin_word(rng);
        (format!("{u} {w}"), format!("{b} {w}"))
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

    /// Vocab growth must read as Japanese: no ASCII digits in any fabricated
    /// text (a digit suffix surfaces in replies as 「…しよう 4999」).
    #[test]
    fn unique_words_are_kana_not_digits() {
        let recs = records(FabricateOpts {
            pairs: 60,
            rng_seed: 2,
            unique_frac: 1.0,
        });
        assert!(recs
            .iter()
            .all(|r| !r.text.chars().any(|c| c.is_ascii_digit())));
        let base: std::collections::HashSet<&str> =
            PAIRS.iter().flat_map(|&(u, b)| [u, b]).collect();
        assert!(
            recs.iter().any(|r| !base.contains(r.text.as_str())),
            "unique_frac=1 must actually grow the surface vocabulary"
        );
    }

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
