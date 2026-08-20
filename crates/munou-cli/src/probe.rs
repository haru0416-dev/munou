//! Load a seed conversation log and compare empty vs grown engines numerically.

use std::fs;
use std::path::{Path, PathBuf};

use std::collections::BTreeSet;

use anyhow::{bail, Context, Result};
use munou_engine::{Engine, OpenConfig, Params, PathKind};

const DEFAULT_PROMPTS: &[&str] = &[
    "おはよう",
    "ありがとう",
    "散歩しない？",
    "コーヒー飲む？",
    "猫みて",
    "眠い",
    "ゲームしない？",
    "量子力学の話をしよう",
];

pub struct ProbeArgs {
    pub seed: PathBuf,
    pub triggers: Option<PathBuf>,
    pub rng_seed: u64,
    pub p_slip: f64,
}

struct Row {
    prompt: String,
    path: PathKind,
    sim: f32,
    band_hit: bool,
    slipped: bool,
    lcs: usize,
    ctx_used: usize,
    gen_len: usize,
    elapsed_us: u128,
    text: String,
    n_sources: usize,
}

struct Run {
    utterances: usize,
    tokens: usize,
    vocab: usize,
    rows: Vec<Row>,
}

fn first_ctx_used(r: &munou_engine::Reply) -> usize {
    r.trace.steps.first().map(|s| s.ctx_len_used).unwrap_or(0)
}

fn collect(engine: &mut Engine, prompts: &[&str]) -> Result<Vec<Row>> {
    let mut rows = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let r = engine.respond(prompt)?;
        rows.push(Row {
            prompt: (*prompt).to_string(),
            path: r.trace.path,
            sim: r.trace.similarity,
            band_hit: r.trace.band_hit,
            slipped: r.trace.slipped,
            lcs: r.trace.novelty_lcs,
            ctx_used: first_ctx_used(&r),
            gen_len: r.trace.chunks.len(),
            elapsed_us: r.trace.elapsed_us,
            text: r.text,
            n_sources: r
                .trace
                .candidates
                .iter()
                .map(|c| c.source)
                .collect::<BTreeSet<_>>()
                .len(),
        });
    }
    Ok(rows)
}

fn open_engine(log: PathBuf, args: &ProbeArgs, params: Params) -> Result<Engine> {
    Engine::open(OpenConfig {
        params,
        seed: args.rng_seed,
        log_path: Some(log),
        triggers_path: args.triggers.clone(),
    })
    .context("open engine")
}

fn params(args: &ProbeArgs) -> Params {
    Params {
        p_slip: args.p_slip.clamp(0.0, 1.0),
        ..Params::default()
    }
}

fn mean(xs: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<f64> = xs.collect();
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn pct(n: usize, d: usize) -> f64 {
    if d == 0 {
        0.0
    } else {
        100.0 * n as f64 / d as f64
    }
}

fn path_tag(p: PathKind) -> &'static str {
    match p {
        PathKind::Trigger => "Trig",
        PathKind::Markov => "Mark",
        PathKind::Retrieve => "Retr",
        PathKind::Echo => "Echo",
    }
}

fn run_on(log: PathBuf, args: &ProbeArgs, prompts: &[&str]) -> Result<Run> {
    let mut engine = open_engine(log, args, params(args))?;
    let st = engine.stats();
    let rows = collect(&mut engine, prompts)?;
    Ok(Run {
        utterances: st.utterances,
        tokens: st.tokens,
        vocab: st.vocab,
        rows,
    })
}

fn copy_seed(src: &Path, dest_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(dest_dir).with_context(|| dest_dir.display().to_string())?;
    let dest = dest_dir.join("log.jsonl");
    fs::copy(src, &dest)
        .with_context(|| format!("copy {} -> {}", src.display(), dest.display()))?;
    Ok(dest)
}

fn tmp(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "munou-probe-{}-{}-{}",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ))
}

pub fn run(args: ProbeArgs) -> Result<()> {
    if !args.seed.exists() {
        bail!("seed log not found: {}", args.seed.display());
    }

    let empty_dir = tmp("empty");
    let seed_dir = tmp("seed");
    let seed_b_dir = tmp("seed-b");
    fs::create_dir_all(&empty_dir)?;
    let empty_log = empty_dir.join("log.jsonl");
    let seed_log = copy_seed(&args.seed, &seed_dir)?;
    let seed_log_b = copy_seed(&args.seed, &seed_b_dir)?;

    let empty = run_on(empty_log, &args, DEFAULT_PROMPTS)?;
    let seeded = run_on(seed_log, &args, DEFAULT_PROMPTS)?;
    let seeded_b = run_on(seed_log_b, &args, DEFAULT_PROMPTS)?;

    println!(
        "seed={}  rng={}  p_slip={:.2}  mix=pool  prompts={}",
        args.seed.display(),
        args.rng_seed,
        args.p_slip,
        DEFAULT_PROMPTS.len()
    );
    println!(
        "corpus  empty: utterances={} tokens={} vocab={}",
        empty.utterances, empty.tokens, empty.vocab
    );
    println!(
        "corpus  seed : utterances={} tokens={} vocab={}",
        seeded.utterances, seeded.tokens, seeded.vocab
    );
    println!();
    println!(
        "{:<16} {:>4} {:>4} {:>4} {:>8} {:>8} {:>8} {:>8}  empty | seeded",
        "prompt", "eP", "sP", "eq", "e_ctx", "s_ctx", "e_sim", "s_sim"
    );

    let mut diverge = 0usize;
    let mut trig_seed = 0usize;
    let mut trig_empty = 0usize;
    let mut band_s = 0usize;
    let mut slip_s = 0usize;
    for (a, b) in empty.rows.iter().zip(seeded.rows.iter()) {
        let eq = a.text == b.text;
        if !eq {
            diverge += 1;
        }
        if a.path == PathKind::Trigger {
            trig_empty += 1;
        }
        if b.path == PathKind::Trigger {
            trig_seed += 1;
        }
        if b.band_hit {
            band_s += 1;
        }
        if b.slipped {
            slip_s += 1;
        }
        println!(
            "{:<16} {:>4} {:>4} {:>4} {:>8} {:>8} {:>8.3} {:>8.3}  {} | {}",
            trunc(&a.prompt, 16),
            path_tag(a.path),
            path_tag(b.path),
            if eq { "Y" } else { "n" },
            a.ctx_used,
            b.ctx_used,
            a.sim,
            b.sim,
            trunc(&a.text, 18),
            trunc(&b.text, 18),
        );
    }

    let n = seeded.rows.len();
    let empty_ctx = mean(empty.rows.iter().map(|r| r.ctx_used as f64));
    let seed_ctx = mean(seeded.rows.iter().map(|r| r.ctx_used as f64));
    let empty_sim = mean(empty.rows.iter().map(|r| r.sim as f64));
    let seed_sim = mean(seeded.rows.iter().map(|r| r.sim as f64));
    let seed_lcs = mean(seeded.rows.iter().map(|r| {
        if r.gen_len == 0 {
            0.0
        } else {
            r.lcs as f64 / r.gen_len as f64
        }
    }));
    let seed_us = mean(seeded.rows.iter().map(|r| r.elapsed_us as f64));
    let max_us = seeded.rows.iter().map(|r| r.elapsed_us).max().unwrap_or(0);

    println!();
    println!(
        "agg  diverge={}/{} ({:.0}%)  trigger empty={:.0}% seed={:.0}%",
        diverge,
        n,
        pct(diverge, n),
        pct(trig_empty, n),
        pct(trig_seed, n)
    );
    println!(
        "agg  mean_ctx empty={:.2} seed={:.2}  mean_sim empty={:.3} seed={:.3}",
        empty_ctx, seed_ctx, empty_sim, seed_sim
    );
    println!(
        "agg  band_hit={:.0}%  slip={:.0}%  rote_lcs={:.2}  mean_us={:.0} max_us={}",
        pct(band_s, n),
        pct(slip_s, n),
        seed_lcs,
        seed_us,
        max_us
    );

    let det_ok = empty.rows.len() == seeded_b.rows.len()
        && seeded
            .rows
            .iter()
            .zip(seeded_b.rows.iter())
            .all(|(a, b)| a.text == b.text && a.path == b.path);

    let ohayo = seeded
        .rows
        .iter()
        .find(|r| r.prompt == "おはよう")
        .expect("probe set includes おはよう");
    let ood = seeded
        .rows
        .iter()
        .find(|r| r.prompt.contains("量子"))
        .expect("probe set includes OOD");

    let mut failed = 0usize;
    let mut check = |name: &str, ok: bool, detail: String| {
        if ok {
            println!("PASS  {name}  {detail}");
        } else {
            failed += 1;
            println!("FAIL  {name}  {detail}");
        }
    };

    check(
        "grow-tokens",
        seeded.tokens > empty.tokens && seeded.tokens >= 40,
        format!("empty={} seed={}", empty.tokens, seeded.tokens),
    );
    check(
        "grow-vocab",
        seeded.vocab > empty.vocab && seeded.vocab >= 16,
        format!("empty={} seed={}", empty.vocab, seeded.vocab),
    );
    check(
        "grow-utterances",
        seeded.utterances == 50,
        format!("seed utterances={} (want 50)", seeded.utterances),
    );
    check(
        "diverge",
        diverge >= 3,
        format!("{diverge}/{n} in-set replies differ empty vs seed"),
    );
    check(
        "ctx-grows",
        seed_ctx + 1e-9 >= empty_ctx,
        format!("mean ctx empty={empty_ctx:.2} seed={seed_ctx:.2}"),
    );
    check(
        "trigger-ohayo",
        ohayo.path == PathKind::Trigger,
        format!("path={:?} text={}", ohayo.path, ohayo.text),
    );
    check(
        "ood-closed",
        ood.path != PathKind::Trigger,
        format!(
            "path={:?} text={} (no trigger for unknown topics)",
            ood.path, ood.text
        ),
    );
    let mixed = seeded
        .rows
        .iter()
        .filter(|r| r.prompt != "おはよう" && r.prompt != "ありがとう")
        .map(|r| r.n_sources)
        .max()
        .unwrap_or(0);
    check(
        "hybrid-pool",
        mixed >= 2,
        format!("max distinct sources on in-domain prompts={mixed}"),
    );
    check(
        "determinism",
        det_ok,
        "same seed log + same rng → identical probe replies".into(),
    );
    if args.p_slip == 0.0 {
        check(
            "zero-slip",
            slip_s == 0,
            format!("slipped {slip_s}/{n} with p_slip=0"),
        );
    }

    let _ = fs::remove_dir_all(&empty_dir);
    let _ = fs::remove_dir_all(&seed_dir);
    let _ = fs::remove_dir_all(&seed_b_dir);

    if failed > 0 {
        bail!("{failed} probe checks failed");
    }
    Ok(())
}

fn trunc(s: &str, max_chars: usize) -> String {
    let n = s.chars().count();
    if n <= max_chars {
        return s.to_string();
    }
    let t: String = s.chars().take(max_chars.saturating_sub(1)).collect();
    format!("{t}…")
}
