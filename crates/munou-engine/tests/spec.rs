//! Design-spec invariants that go beyond per-module unit tests.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use munou_engine::{Engine, MixMode, OpenConfig, Params, PathKind, TriggerDict};

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
fn trigger_wins_greeting_against_echo() {
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
    assert!(
        r.trace
            .candidates
            .iter()
            .any(|c| c.source == PathKind::Echo),
        "pool should still list echo, not XOR it away"
    );
    let _ = fs::remove_dir_all(&dir);
}
