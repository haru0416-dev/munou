//! Design-spec invariants that go beyond per-module unit tests.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use munou_engine::{Engine, OpenConfig, Params, PathKind, TriggerDict};

fn tmp_dir(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "munou-spec-{}-{}-{}",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn trigger_path_is_exclusive() {
    let dir = tmp_dir("trig");
    let trig = dir.join("t.json");
    fs::write(
        &trig,
        r#"[{"pattern":"おはよう","responses":["おはよ・テスト応答"]}]"#,
    )
    .unwrap();
    let mut e = Engine::ephemeral(Params::default(), 3).unwrap();
    e.load_triggers(&trig).unwrap();
    let r = e.respond("おはよう").unwrap();
    assert_eq!(r.trace.path, PathKind::Trigger);
    assert_eq!(r.text, "おはよ・テスト応答");
    assert!(r.trace.trigger.is_some());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn zero_slip_never_selects_below_rank_one() {
    let params = Params {
        p_slip: 0.0,
        n_cand: 8,
        ..Params::default()
    };
    let mut e = Engine::ephemeral(params, 11).unwrap();
    for line in ["こんにちは", "今日は晴れ", "散歩しよう", "ね"] {
        let r = e.respond(line).unwrap();
        assert!(!r.trace.slipped, "slipped with p_slip=0");
        assert_eq!(r.trace.chosen_rank, 0);
    }
}

#[test]
fn full_slip_picks_non_top_when_multiple_candidates() {
    let params = Params {
        p_slip: 1.0,
        n_cand: 8,
        ..Params::default()
    };
    let mut e = Engine::ephemeral(params, 11).unwrap();
    let mut slipped = false;
    for line in [
        "こんにちは世界",
        "今日はとても良い天気ですね",
        "また明日会いましょう",
    ] {
        let r = e.respond(line).unwrap();
        if r.trace.candidates.len() >= 2 && r.trace.slipped {
            slipped = true;
            assert!(r.trace.chosen_rank >= 1);
        }
    }
    assert!(slipped, "p_slip=1 should slip when ≥2 candidates exist");
}

#[test]
fn explain_chain_is_complete() {
    let mut e = Engine::ephemeral(Params::default(), 4).unwrap();
    let r = e.respond("説明可能性のテスト").unwrap();
    assert_eq!(r.trace.input, "説明可能性のテスト");
    assert!(!r.trace.morphemes.is_empty());
    assert!(!r.trace.chunks.is_empty());
    assert!(!r.trace.candidates.is_empty());
    assert!(r.trace.candidates.iter().any(|c| c.chosen));
    assert!(r.trace.elapsed_us > 0);
}

#[test]
fn jsonl_log_is_source_of_truth() {
    let dir = tmp_dir("log");
    let log = dir.join("log.jsonl");
    {
        let mut e = Engine::open(OpenConfig {
            params: Params::default(),
            seed: 5,
            log_path: Some(log.clone()),
            triggers_path: None,
        })
        .unwrap();
        e.respond("第一声").unwrap();
        e.respond("第二声").unwrap();
    }
    let raw = fs::read_to_string(&log).unwrap();
    let lines: Vec<&str> = raw.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(lines.len(), 4, "user+bot × 2 turns");
    for line in &lines {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(v["v"], 1);
        assert!(v["text"].as_str().unwrap().chars().count() > 0);
    }
    let e2 = Engine::open(OpenConfig {
        params: Params::default(),
        seed: 5,
        log_path: Some(log.clone()),
        triggers_path: None,
    })
    .unwrap();
    assert_eq!(e2.stats().utterances, 4);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn kn_smoothing_responds() {
    let params = Params {
        smoothing: munou_engine::SmoothingKind::Kn,
        ..Params::default()
    };
    let mut e = Engine::ephemeral(params, 8).unwrap();
    let r = e.respond("ナイーブではない平滑化").unwrap();
    assert!(!r.text.is_empty());
}

#[test]
fn trigger_dict_parses_example() {
    let p =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/triggers.example.json");
    let dict = TriggerDict::from_path(&p).expect("example triggers");
    assert!(!dict.is_empty());
}

#[test]
fn user_chunks_enter_generation_context() {
    let mut e = Engine::ephemeral(Params::default(), 2).unwrap();
    let r = e.respond("これは文脈に入るはず").unwrap();
    if r.trace.path == PathKind::Markov && !r.trace.steps.is_empty() {
        assert!(
            r.trace.steps[0].ctx_len_requested >= r.trace.chunks.len(),
            "requested={} chunks={}",
            r.trace.steps[0].ctx_len_requested,
            r.trace.chunks.len()
        );
    }
}

#[test]
fn cargo_lock_has_no_network_or_llm_crates() {
    let lock = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.lock"));
    let banned = [
        "async-openai",
        "burn",
        "candle-core",
        "hf-hub",
        "hyper",
        "llama-cpp-2",
        "llm",
        "reqwest",
        "tch",
        "tokenizers",
        "tokio",
        "tract-core",
        "ureq",
    ];
    for line in lock.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("name = \"") else {
            continue;
        };
        let Some(name) = rest.strip_suffix('"') else {
            continue;
        };
        assert!(
            !banned.contains(&name),
            "closed-world lockfile must not include {name}"
        );
    }
}

#[test]
fn append_survives_partial_process_exit() {
    let dir = tmp_dir("fsync");
    let log = dir.join("log.jsonl");
    let mut e = Engine::open(OpenConfig {
        params: Params::default(),
        seed: 1,
        log_path: Some(log.clone()),
        triggers_path: None,
    })
    .unwrap();
    e.respond("落ちても残る").unwrap();
    drop(e);
    // Simulate a new process: the JSONL must already be complete on disk.
    let mut f = fs::OpenOptions::new().append(true).open(&log).unwrap();
    f.write_all(b"").unwrap();
    f.sync_all().unwrap();
    let e2 = Engine::open(OpenConfig {
        params: Params::default(),
        seed: 1,
        log_path: Some(log.clone()),
        triggers_path: None,
    })
    .unwrap();
    assert!(e2.stats().utterances >= 2);
    let _ = fs::remove_dir_all(&dir);
}
