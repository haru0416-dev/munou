//! Usage soak: weird inputs, param corners, log damage, reopen.
//! Failures here are panics or broken invariants, not "unfunny" replies.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use munou_engine::{
    fabricate_write, Engine, FabricateOpts, MixMode, OpenConfig, Params, PathKind, SmoothingKind,
    Stage, TriggerDict,
};

fn tmp(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "munou-soak-{}-{}-{}",
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

fn open_log(dir: &std::path::Path, params: Params, seed: u64) -> Engine {
    Engine::open(OpenConfig {
        params,
        seed,
        log_path: Some(dir.join("log.jsonl")),
        triggers_path: None,
    })
    .unwrap()
}

#[test]
fn never_panics_on_weird_inputs() {
    let mut e = Engine::ephemeral(Params::default(), 1).unwrap();
    let long_ja = "あ".repeat(400);
    let long_ascii = "A".repeat(800);
    let inputs = vec![
        "",
        "   ",
        "\t\n",
        "a",
        "あ",
        "😊",
        "👨‍👩‍👧‍👦",
        "hello世界123",
        "‏RTL ערב",
        "ゼロ幅\u{200b}スペース",
        "「引用」と『二重』",
        "??!!。。、、",
        "http://example.invalid/foo?x=1",
        "line1\nline2",
        "null\0inside",
        long_ja.as_str(),
        long_ascii.as_str(),
        "<script>alert(1)</script>",
        "{\"role\":\"user\"}",
        "/good",
        "/observe",
        "量子力学の話をしよう",
        "おはよう",
        "　　全角空白　　",
    ];
    for (i, s) in inputs.iter().enumerate() {
        let r = e
            .respond(s)
            .unwrap_or_else(|err| panic!("respond[{i}] {s:?}: {err}"));
        assert!(
            !r.text.is_empty() || s.trim().is_empty(),
            "empty reply for non-empty input {s:?}"
        );
        assert!(r.trace.candidates.iter().any(|c| c.chosen) || r.trace.candidates.is_empty());
        let _ = e.observe().panel();
        let _ = e.observe().html();
        let _ = e.observe().strip();
        let _ = r.trace.explain_text();
    }
}

#[test]
fn empty_and_whitespace_do_not_crash_and_are_named() {
    let mut e = Engine::ephemeral(Params::default(), 2).unwrap();
    let a = e.respond("").unwrap();
    let b = e.respond("   \t").unwrap();
    assert!(
        !a.text.is_empty(),
        "empty input should still emit a fallback"
    );
    assert!(!b.text.is_empty());
    assert!(e.stats().utterances >= 4);
}

#[test]
fn param_corners_respond() {
    let corners = [
        Params {
            n_cand: 1,
            n_retrieve: 0,
            n_echo: 1,
            p_slip: 0.0,
            p_learn: 0.0,
            tau_gen: 1e-3,
            tau_slip: 1e-3,
            p_nucleus: 0.0,
            k_top: 1,
            max_gen_len: 1,
            merge_threshold: 32,
            ..Params::default()
        },
        Params {
            n_cand: 32,
            n_retrieve: 12,
            n_echo: 2,
            p_slip: 1.0,
            p_learn: 1.0,
            tau_gen: 2.5,
            p_nucleus: 0.5,
            k_top: 3,
            smoothing: SmoothingKind::Kn,
            kn_discount: 0.0,
            mix: MixMode::Pool,
            mmr_lambda: 0.0,
            band_penalty: 0.0,
            rote_penalty: 0.0,
            ..Params::default()
        },
        Params {
            mix: MixMode::Exclusive,
            p_learn: 1.0,
            p_slip: 0.0,
            n_cand: 4,
            ..Params::default()
        },
        Params {
            theta_trig: 0.0,
            pref_step: 1.0,
            pref_clip: 0.01,
            embed_dim: 8,
            k_topic: 1,
            l_max: 1,
            chunk_morphs: 1,
            entropy_n: 2,
            ..Params::default()
        },
    ];
    for (i, p) in corners.into_iter().enumerate() {
        let mut e = Engine::ephemeral(p, 11 + i as u64).unwrap();
        for line in ["こんにちは", "今日は晴れ", "ね", "量子"] {
            let r = e.respond(line).unwrap();
            assert!(!r.text.is_empty(), "corner {i} empty reply");
        }
        e.rebuild().unwrap();
        e.retokenize_from_log().unwrap();
        e.respond("再分割のあと").unwrap();
        let _ = e.feedback(true);
        let _ = e.feedback(false);
        let _ = e.feedback(true);
    }
}

#[test]
fn damaged_jsonl_is_skipped_not_fatal() {
    let dir = tmp("damaged");
    let log = dir.join("log.jsonl");
    let mut f = fs::File::create(&log).unwrap();
    writeln!(f, "{{not json").unwrap();
    writeln!(f).unwrap();
    writeln!(
        f,
        r#"{{"v":1,"t":1,"role":"user","text":"残ってほしい","learned":true}}"#
    )
    .unwrap();
    writeln!(
        f,
        r#"{{"v":1,"t":2,"role":"bot","text":"了解","learned":true}}"#
    )
    .unwrap();
    writeln!(f, r#"{{"v":1,"t":3,"role":"system","text":"skip me"}}"#).unwrap();
    writeln!(f, r#"{{"v":1,"t":4,"role":"META","text":"good"}}"#).unwrap();
    drop(f);
    let mut e = open_log(&dir, Params::default(), 3);
    assert!(e.stats().utterances >= 2);
    let r = e.respond("続き").unwrap();
    assert!(!r.text.is_empty());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn meta_then_reopen_does_not_inflate_seed() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = tmp("seed-meta");
    fs::copy(root.join("data/seed.jsonl"), dir.join("log.jsonl")).unwrap();
    let params = Params {
        p_learn: 1.0,
        p_slip: 0.0,
        ..Params::default()
    };
    {
        let mut e = open_log(&dir, params.clone(), 1);
        assert_eq!(e.stats().utterances, 50);
        assert_eq!(e.stats().learned, 50);
        e.respond("コーヒー飲む？").unwrap();
        e.feedback(true).unwrap();
        e.feedback(false).unwrap();
        assert_eq!(e.stats().utterances, 52);
        assert_eq!(e.stats().meta, 2);
        assert_eq!(e.observe().stage, Stage::Growing);
    }
    let e2 = open_log(&dir, params, 1);
    assert_eq!(e2.stats().utterances, 52);
    assert_eq!(e2.stats().learned, 52);
    assert_eq!(e2.stats().meta, 2);
    assert_eq!(e2.stats().utterances + e2.stats().meta, 54);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn good_after_reopen_uses_last_bot_path() {
    let dir = tmp("pref-reopen");
    let params = Params {
        p_learn: 1.0,
        p_slip: 0.0,
        ..Params::default()
    };
    let path;
    {
        let mut e = open_log(&dir, params.clone(), 8);
        let r = e.respond("ありがとう").unwrap();
        path = r.trace.path;
        assert_ne!(e.feedback(true).unwrap(), "no turn yet");
    }
    let mut e2 = open_log(&dir, params, 8);
    let msg = e2.feedback(true).unwrap();
    assert_ne!(
        msg, "no turn yet",
        "reopen must remember last path for /good"
    );
    assert!(
        e2.stats().path_prior[path_index(path)] > 0.0,
        "prior should accumulate across reopen: {msg}"
    );
    let _ = fs::remove_dir_all(&dir);
}

fn path_index(p: PathKind) -> usize {
    match p {
        PathKind::Trigger => 0,
        PathKind::Markov => 1,
        PathKind::Retrieve => 2,
        PathKind::Echo => 3,
        PathKind::Adapt => 4,
    }
}

#[test]
fn trigger_dict_edges() {
    let dir = tmp("trig");
    fs::write(dir.join("empty.json"), "[]").unwrap();
    fs::write(
        dir.join("blank.json"),
        r#"[{"pattern":"x","responses":[]},{"pattern":"","responses":["y"]}]"#,
    )
    .unwrap();
    assert!(TriggerDict::from_path(&dir.join("empty.json"))
        .unwrap()
        .is_empty());
    let mut e = Engine::ephemeral(Params::default(), 1).unwrap();
    e.load_triggers(&dir.join("blank.json")).unwrap();
    let r = e.respond("x").unwrap();
    assert!(!r.text.is_empty());
    let r2 = e.respond("").unwrap();
    assert!(!r2.text.is_empty());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn observe_html_escapes_and_memory_line() {
    let mut e = Engine::ephemeral(
        Params {
            p_learn: 1.0,
            ..Params::default()
        },
        4,
    )
    .unwrap();
    e.respond("<b>x</b>").unwrap();
    let h = e.observe().html();
    assert!(h.contains("charset=\"utf-8\""));
    assert!(h.contains("記憶"));
    assert!(!h.contains("<b>x</b>"));
    let p = e.observe().panel();
    assert!(p.contains("記憶"));
}

#[test]
fn kn_and_exclusive_grow_from_seed() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = tmp("kn-seed");
    fs::copy(root.join("data/seed.jsonl"), dir.join("log.jsonl")).unwrap();
    let params = Params {
        smoothing: SmoothingKind::Kn,
        mix: MixMode::Exclusive,
        p_slip: 0.0,
        p_learn: 1.0,
        ..Params::default()
    };
    let mut e = Engine::open(OpenConfig {
        params,
        seed: 1,
        log_path: Some(dir.join("log.jsonl")),
        triggers_path: Some(root.join("data/triggers.example.json")),
    })
    .unwrap();
    assert_eq!(e.stats().utterances, 50);
    let hi = e.respond("おはよう").unwrap();
    assert_eq!(hi.trace.path, PathKind::Trigger);
    let ood = e.respond("量子力学の話をしよう").unwrap();
    assert_ne!(ood.trace.path, PathKind::Trigger);
    e.retokenize_from_log().unwrap();
    assert!(e.stats().tokens > 0);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn nucleus_does_not_kill_interpolation_when_off() {
    let params = Params {
        p_learn: 1.0,
        p_slip: 0.0,
        p_nucleus: 1.0,
        k_top: 0,
        mix: MixMode::Exclusive,
        ..Params::default()
    };
    let mut e = Engine::ephemeral(params, 2).unwrap();
    for line in [
        "赤青黄",
        "赤青黄",
        "赤青黄",
        "赤青緑",
        "赤青緑",
        "赤青緑",
        "赤青",
    ] {
        e.respond(line).unwrap();
    }
    let r = e.respond("赤青").unwrap();
    assert!(!r.text.is_empty());
}

#[test]
fn many_turns_stay_fast_and_deterministic() {
    let params = Params {
        p_learn: 1.0,
        p_slip: 0.0,
        merge_threshold: 64,
        n_cand: 6,
        ..Params::default()
    };
    let prompts = [
        "こんにちは",
        "今日はいい天気",
        "コーヒー飲む？",
        "猫かわいい",
        "ゲームしない？",
        "ねむい",
        "また今度",
        "ありがとう",
    ];
    let t0 = Instant::now();
    let mut a = Engine::ephemeral(params.clone(), 99).unwrap();
    let mut b = Engine::ephemeral(params, 99).unwrap();
    let mut texts = Vec::new();
    for i in 0..40 {
        let line = prompts[i % prompts.len()];
        let ra = a.respond(line).unwrap();
        let rb = b.respond(line).unwrap();
        assert_eq!(ra.text, rb.text, "turn {i}");
        texts.push(ra.text);
        assert!(
            ra.trace.elapsed_us < 50_000,
            "slow turn {}us",
            ra.trace.elapsed_us
        );
    }
    let dt = t0.elapsed();
    assert!(dt.as_millis() < 8_000, "40+40 turns took {dt:?}");
    a.rebuild().unwrap();
    assert!(a.stats().tokens > 0);
    assert_eq!(texts.len(), 40);
}

#[test]
fn chatty_unicode_roundtrip_log() {
    let dir = tmp("uni");
    let params = Params {
        p_learn: 1.0,
        ..Params::default()
    };
    {
        let mut e = open_log(&dir, params.clone(), 5);
        e.respond("絵文字😊と漢字とカタカナミックス").unwrap();
        e.respond("café naïve").unwrap();
    }
    let raw = fs::read_to_string(dir.join("log.jsonl")).unwrap();
    assert!(raw.contains("😊"));
    let e2 = open_log(&dir, params, 5);
    assert!(e2.stats().utterances >= 4);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn why_after_reopen_uses_last_bot_line() {
    let dir = tmp("why-reopen");
    let params = Params {
        p_learn: 1.0,
        p_slip: 0.0,
        ..Params::default()
    };
    {
        let mut e = open_log(&dir, params.clone(), 3);
        e.respond("こんにちは").unwrap();
        assert!(e.why_text().contains("path="));
        assert!(!e.why_text().contains("reopened"));
    }
    let e2 = open_log(&dir, params, 3);
    let why = e2.why_text();
    assert!(!why.contains("no trace yet"), "{why}");
    assert!(why.contains("reopened") || why.contains("path="), "{why}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn seed_live_latency_stays_in_budget() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = tmp("lat");
    fs::copy(root.join("data/seed.jsonl"), dir.join("log.jsonl")).unwrap();
    let mut e = Engine::open(OpenConfig {
        params: Params {
            p_learn: 1.0,
            p_slip: 0.0,
            ..Params::default()
        },
        seed: 1,
        log_path: Some(dir.join("log.jsonl")),
        triggers_path: Some(root.join("data/triggers.example.json")),
    })
    .unwrap();
    let prompts = [
        "散歩しない？",
        "コーヒー飲む？",
        "猫みて",
        "ゲームしない？",
        "量子力学の話をしよう",
        "おはよう",
        "ねむい",
        "ありがとう",
    ];
    let mut times = Vec::new();
    let t0 = Instant::now();
    for i in 0..80 {
        let r = e.respond(prompts[i % prompts.len()]).unwrap();
        times.push(r.trace.elapsed_us);
        assert!(!r.text.is_empty());
    }
    let wall = t0.elapsed();
    times.sort_unstable();
    let p99 = times[(times.len() * 99 / 100).min(times.len() - 1)];
    // 30ms is a release NFR (verify skips it in debug); debug + parallel
    // test binaries only get a sanity bound.
    let budget: u128 = if cfg!(debug_assertions) {
        120_000
    } else {
        30_000
    };
    assert!(
        p99 < budget,
        "p99={p99}us wall={wall:?} (design 30ms incl. embed; debug bound {budget}us)"
    );
    assert!(wall.as_millis() < 5_000, "80 turns wall {wall:?}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn inf_jsonl_score_does_not_poison_eval() {
    let dir = tmp("nan");
    let log = dir.join("log.jsonl");
    let mut f = fs::File::create(&log).unwrap();
    writeln!(
        f,
        r#"{{"v":1,"t":1,"role":"user","text":"hi","learned":true}}"#
    )
    .unwrap();
    writeln!(
        f,
        r#"{{"v":1,"t":2,"role":"bot","text":"yo","learned":true,"score":null}}"#
    )
    .unwrap();
    writeln!(
        f,
        r#"{{"v":1,"t":3,"role":"bot","text":"bad","learned":true,"score":"NaN"}}"#
    )
    .unwrap();
    drop(f);
    let e = open_log(&dir, Params::default(), 1);
    let s = e.eval_summary();
    assert!(!s.to_lowercase().contains("nan"), "{s}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn fabricated_thousands_are_dense_and_closed() {
    let dir = tmp("fab");
    let log = dir.join("log.jsonl");
    let n = fabricate_write(
        &log,
        FabricateOpts {
            pairs: 400,
            rng_seed: 1,
            unique_frac: 0.0,
        },
    )
    .unwrap();
    assert_eq!(n, 800);
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut e = Engine::open(OpenConfig {
        params: Params {
            p_slip: 0.0,
            ..Params::default()
        },
        seed: 1,
        log_path: Some(log),
        triggers_path: Some(root.join("data/triggers.example.json")),
    })
    .unwrap();
    let st = e.stats();
    assert_eq!(st.utterances, 800);
    assert_eq!(st.learned, 800);
    assert!(st.tokens >= 400, "tokens={}", st.tokens);
    assert_eq!(e.observe().stage, Stage::Dense);
    let hi = e.respond("おはよう").unwrap();
    assert_eq!(hi.trace.path, PathKind::Trigger);
    let ood = e.respond("量子力学の話をしよう").unwrap();
    assert_ne!(ood.trace.path, PathKind::Trigger);
    assert!(e.observe().panel().contains("濃い"));
    let _ = fs::remove_dir_all(&dir);
}
