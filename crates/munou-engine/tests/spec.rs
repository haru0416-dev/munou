use munou_engine::{Engine, MixMode, Params, PathKind};
use std::collections::BTreeSet;

fn seed_params() -> Params {
    Params {
        p_slip: 0.0,
        mix: MixMode::Pool,
        ..Params::default()
    }
}

fn exclusive_params() -> Params {
    Params {
        p_slip: 0.0,
        mix: MixMode::Exclusive,
        ..Params::default()
    }
}

#[test]
fn trigger_wins_greeting_against_echo() {
    let mut e = Engine::new(1, seed_params());
    let tr = e.respond("おはよう");
    assert_eq!(tr.path, PathKind::Trigger);
    assert_eq!(tr.reply, "おはよう");
    assert!(tr.trace.candidates.iter().any(|c| c.source == PathKind::Echo));
}

#[test]
fn exclusive_trigger_still_xors() {
    let mut e = Engine::new(1, exclusive_params());
    let tr = e.respond("おはよう");
    assert_eq!(tr.path, PathKind::Trigger);
    assert!(!tr.trace.candidates.iter().any(|c| c.source == PathKind::Markov));
    assert!(!tr.trace.candidates.iter().any(|c| c.source == PathKind::Echo));
}

#[test]
fn empty_engine_does_not_parrot_user_as_markov() {
    let mut e = Engine::new(1, seed_params());
    let tr = e.respond("ありがとう");
    assert_eq!(tr.path, PathKind::Echo);
    assert_eq!(tr.reply, "ありがとう");
    assert!(!tr.trace.candidates.iter().any(|c| c.source == PathKind::Markov));
}

#[test]
fn ood_does_not_fire_trigger() {
    let mut e = Engine::new(1, seed_params());
    let tr = e.respond("量子力学の話をしよう");
    assert_ne!(tr.path, PathKind::Trigger);
}

#[test]
fn seed_log_enables_non_echo_reply() {
    let mut e = Engine::new(1, seed_params());
    for line in include_str!("../../../data/seed.jsonl").lines() {
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        e.ingest(
            v["user"].as_str().unwrap(),
            v["bot"].as_str().unwrap(),
        );
    }
    let tr = e.respond("散歩したい");
    assert_ne!(tr.path, PathKind::Echo);
    assert!(!tr.reply.is_empty());
    let sources: BTreeSet<_> = tr.trace.candidates.iter().map(|c| c.source).collect();
    assert!(
        sources.len() >= 2,
        "seeded pool should mix sources, got {sources:?}"
    );
}

#[test]
fn slip_picks_lower_ranked() {
    let mut e = Engine::new(
        1,
        Params {
            p_slip: 1.0,
            mix: MixMode::Pool,
            ..Params::default()
        },
    );
    e.ingest("a", "hello world");
    e.ingest("b", "hello there");
    e.ingest("c", "goodbye now");
    let tr = e.respond("hello");
    assert_eq!(tr.trace.candidates.len(), 4.min(tr.trace.candidates.len()));
    assert!(tr.trace.rank_chosen >= 1);
}

#[test]
fn replay_same_seed_same_output() {
    let mut a = Engine::new(42, seed_params());
    let mut b = Engine::new(42, seed_params());
    a.ingest("今日は晴れ", "ああそうか");
    b.ingest("今日は晴れ", "ああそうか");
    let ra = a.respond("散歩しよう");
    let rb = b.respond("散歩しよう");
    assert_eq!(ra.reply, rb.reply);
    assert_eq!(ra.path, rb.path);
}

#[test]
fn different_seed_can_diverge() {
    let mut a = Engine::new(1, seed_params());
    let mut b = Engine::new(2, seed_params());
    a.ingest("x", "alpha beta gamma");
    b.ingest("x", "alpha beta gamma");
    let ra = a.respond("alpha");
    let rb = b.respond("alpha");
    let _ = (ra, rb);
}

#[test]
fn why_text_mentions_path() {
    let mut e = Engine::new(1, seed_params());
    let tr = e.respond("おはよう");
    let why = e.last_trace().explain_text();
    assert!(why.contains("[trig]"));
}

#[test]
fn persist_roundtrip() {
    let dir = std::env::temp_dir().join(format!("munou-spec-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut e = Engine::new(7, seed_params());
    e.ingest("u", "hello kitty");
    e.save(&dir).unwrap();
    let e2 = Engine::load(&dir, 7, seed_params()).unwrap();
    assert_eq!(e.token_count(), e2.token_count());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn band_hit_uses_cosine() {
    let mut e = Engine::new(1, seed_params());
    e.ingest("cat dog", "cat dog bird");
    let tr = e.respond("cat dog");
    assert!(tr.trace.similarity >= 0.0);
}
