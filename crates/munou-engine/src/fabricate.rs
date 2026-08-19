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
