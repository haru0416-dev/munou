//! Fabricate a large closed conversation log and time open + respond.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use munou_engine::{Engine, FabricateOpts, OpenConfig, Params};

const PROMPTS: &[&str] = &[
    "おはよう",
    "ありがとう",
    "散歩しない？",
    "コーヒー飲む？",
    "猫みて",
    "ゲームしない？",
    "眠い",
    "量子力学の話をしよう",
];

pub struct ScaleArgs {
    pub pairs: usize,
    pub rng_seed: u64,
    pub unique_frac: f64,
    pub out: Option<PathBuf>,
    pub p_slip: f64,
    pub p_learn: f64,
}

pub fn run(args: ScaleArgs) -> Result<()> {
    let pairs = args.pairs.max(1);
    let log = match &args.out {
        Some(p) if p.extension().and_then(|s| s.to_str()) == Some("jsonl") => {
            if let Some(parent) = p.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)
                        .with_context(|| parent.display().to_string())?;
                }
            }
            p.clone()
        }
        Some(d) => {
            std::fs::create_dir_all(d).with_context(|| d.display().to_string())?;
            d.join("log.jsonl")
        }
        None => {
            let d = std::env::temp_dir().join(format!(
                "munou-scale-{}-{}",
                std::process::id(),
                args.rng_seed
            ));
            std::fs::create_dir_all(&d)?;
            d.join("log.jsonl")
        }
    };

    let n = munou_engine::fabricate_write(
        &log,
        FabricateOpts {
            pairs,
            rng_seed: args.rng_seed,
            unique_frac: args.unique_frac.clamp(0.0, 1.0),
        },
    )?;
    let bytes = std::fs::metadata(&log)?.len();
    println!(
        "fabricate  pairs={}  utterances={}  bytes={}  path={}",
        pairs,
        n,
        bytes,
        log.display()
    );

    let params = Params {
        p_slip: args.p_slip.clamp(0.0, 1.0),
        p_learn: args.p_learn.clamp(0.0, 1.0),
        ..Params::default()
    };
    let t0 = Instant::now();
    let mut engine = Engine::open(OpenConfig {
        params,
        seed: args.rng_seed,
        log_path: Some(log.clone()),
        triggers_path: {
            let p = PathBuf::from("data/triggers.example.json");
            p.exists().then_some(p)
        },
    })?;
    let open_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let st = engine.stats();
    let obs = engine.observe();
    println!(
        "open      {:.1}ms  utterances={} learned={} tokens={} vocab={} stage={}",
        open_ms,
        st.utterances,
        st.learned,
        st.tokens,
        st.vocab,
        obs.stage.label()
    );

    let mut times = Vec::new();
    println!(
        "{:<16} {:>6} {:>8} {:>8}  reply",
        "prompt", "path", "sim", "us"
    );
    for p in PROMPTS {
        let r = engine.respond(p)?;
        times.push(r.trace.elapsed_us);
        println!(
            "{:<16} {:>6} {:>8.3} {:>8}  {}",
            trunc(p, 16),
            r.trace.path.tag(),
            r.trace.similarity,
            r.trace.elapsed_us,
            trunc(&r.text, 24)
        );
    }
    times.sort_unstable();
    let p50 = times[times.len() / 2];
    let p99 = times[(times.len() * 99 / 100).min(times.len() - 1)];
    let max = *times.last().unwrap_or(&0);
    println!(
        "respond   p50={p50}us  p99={p99}us  max={max}us  n={}",
        times.len()
    );
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
