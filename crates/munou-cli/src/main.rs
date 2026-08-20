use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use munou_engine::{Engine, MixMode, OpenConfig, Params, SmoothingKind};

mod probe;
mod scale;
mod verify;

#[derive(Parser, Debug)]
#[command(
    name = "munou",
    about = "人工無脳君 — 観察できる育成。LLM を使わない対話エンジン"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Interactive REPL.
    Chat {
        #[command(flatten)]
        common: Common,
        /// Print a generation/selection trace after every turn.
        #[arg(long)]
        explain: bool,
    },
    /// One-shot reply (non-interactive).
    Say {
        #[command(flatten)]
        common: Common,
        text: String,
        #[arg(long)]
        explain: bool,
    },
    /// Print corpus / vocab stats.
    Stats {
        #[command(flatten)]
        common: Common,
    },
    /// Print the raising window (gauges from log + eval). Adapters are out of scope.
    Observe {
        #[command(flatten)]
        common: Common,
        /// `text` (default) or `html` (self-contained; stdout; no server).
        #[arg(long, default_value = "text")]
        format: String,
    },
    /// Print あゆみ — the individual's record: birth day, 初語, 節目, 日和, 関心.
    Ayumi {
        #[command(flatten)]
        common: Common,
    },
    /// Force a generation-buffer merge + SA rebuild.
    Rebuild {
        #[command(flatten)]
        common: Common,
    },
    /// Retokenize the whole log with the current entropy model, then rebuild.
    Retokenize {
        #[command(flatten)]
        common: Common,
    },
    /// Microbench suffix-array construction (synthetic tokens).
    Bench {
        #[arg(long, default_value_t = 1_000_000)]
        tokens: usize,
    },
    /// Run design-spec checks (determinism, trigger, slip, latency, SA).
    Verify {
        #[arg(long, default_value_t = 1_000_000)]
        sa_tokens: usize,
        #[arg(long, default_value_t = 200)]
        turns: usize,
    },
    /// Load a seed conversation log and print empty-vs-grown numbers.
    Probe {
        /// Seed JSONL (`role` user/bot records). Copied to a temp dir; not mutated.
        #[arg(long, default_value = "data/seed.jsonl")]
        seed: PathBuf,
        /// Trigger dictionary. Defaults to `data/triggers.example.json` if present.
        #[arg(long)]
        triggers: Option<PathBuf>,
        /// RNG seed. Same log + same seed → identical probe replies.
        #[arg(long, default_value_t = 1)]
        rng_seed: u64,
        /// Slip injection probability (0 keeps the table readable).
        #[arg(long, default_value_t = 0.0)]
        p_slip: f64,
        /// Live absorb probability. Probe defaults to 1 so empty-vs-seed is about the log.
        #[arg(long, default_value_t = 1.0)]
        p_learn: f64,
    },
    /// Write a large closed conversation log (seed themes, not an external corpus).
    Fabricate {
        /// User/bot pairs. Utterances = 2 × pairs.
        #[arg(long, default_value_t = 10_000)]
        pairs: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// 0 recycles the seed register (huge count, small vocab). >0 appends turn ids.
        #[arg(long, default_value_t = 0.0)]
        unique_frac: f64,
        /// Output JSONL path.
        #[arg(long, default_value = "munou-data/grown.jsonl")]
        out: PathBuf,
    },
    /// Fabricate a huge log, time `Engine::open`, then respond to a fixed prompt set.
    Scale {
        #[arg(long, default_value_t = 10_000)]
        pairs: usize,
        #[arg(long, default_value_t = 1)]
        rng_seed: u64,
        #[arg(long, default_value_t = 0.0)]
        unique_frac: f64,
        /// Directory or `.jsonl` path. Default is a temp dir.
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, default_value_t = 0.0)]
        p_slip: f64,
        #[arg(long, default_value_t = 1.0)]
        p_learn: f64,
    },
}

#[derive(clap::Args, Debug, Clone)]
struct Common {
    /// Directory for the append-only log (`log.jsonl`).
    #[arg(long, env = "MUNOU_DATA", default_value = "./munou-data")]
    data_dir: PathBuf,
    /// Optional trigger dictionary (JSON array of {pattern, responses}).
    #[arg(long)]
    triggers: Option<PathBuf>,
    /// RNG seed. Same log + same seed → identical replies.
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Markov candidate count.
    #[arg(long)]
    n_cand: Option<usize>,
    /// Slip injection probability.
    #[arg(long)]
    p_slip: Option<f64>,
    /// Probability of absorbing a live turn into the corpus (log always appends).
    #[arg(long)]
    p_learn: Option<f64>,
    /// Generation temperature.
    #[arg(long)]
    tau: Option<f64>,
    /// Slip Boltzmann temperature (ranks ≥ 2).
    #[arg(long)]
    tau_slip: Option<f64>,
    /// Band-hinge weight on the selector.
    #[arg(long)]
    band_penalty: Option<f32>,
    /// Penalty × overlap with the bot's own recent replies. 0 disables.
    #[arg(long)]
    self_penalty: Option<f32>,
    /// Disable keyword-anchored bidirectional generation (and its reversed
    /// twin store).
    #[arg(long)]
    no_bidir: bool,
    /// Max Adapt proposals (0 disables; 2 adds a quoted past user line).
    #[arg(long)]
    n_adapt: Option<usize>,
    /// Experimental: selection weight on generation surprise (mean −ln p).
    #[arg(long)]
    surprise_weight: Option<f32>,
    /// Smoothing: naive (Witten-Bell) | kn (modified Kneser-Ney)
    #[arg(long)]
    smoothing: Option<String>,
    /// Skip-gram mix weight on sparse contexts. 0 disables.
    #[arg(long)]
    lambda_skip: Option<f64>,
    /// Recency-cache mix weight on sparse contexts. 0 disables.
    #[arg(long)]
    lambda_cache: Option<f64>,
    /// PPM-C exclusion for Witten-Bell (modified KN already excludes).
    #[arg(long)]
    ppm: bool,
    /// Candidate mix: pool (default) | exclusive (v0.1 XOR)
    #[arg(long)]
    mix: Option<String>,
    /// MMR λ for retrieve: λ·sim − (1−λ)·max redundancy. 1 = top-k cosine.
    #[arg(long)]
    mmr: Option<f32>,
    /// Nucleus mass (LLM top-p analog). 1 = off (default; keeps interpolation tails).
    #[arg(long, alias = "top-p")]
    p_nucleus: Option<f64>,
    /// Decode top-k after nucleus. 0 = off.
    #[arg(long, alias = "top-k")]
    k_top: Option<usize>,
    /// Recent bot utterances scanned for retrieve. 0 = all. Default 1024.
    #[arg(long)]
    retrieve_scan: Option<usize>,
    /// Disable 日和 (deterministic per-day modulation; every day becomes なぎ).
    #[arg(long)]
    no_weather: bool,
    /// 合いの手 base probability per reply. 0 disables it.
    #[arg(long)]
    interject_rate: Option<f64>,
    /// 関心 selection weight (dual-timescale chunk interest). 0 disables.
    #[arg(long)]
    interest_weight: Option<f32>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Chat { common, explain } => chat(common, explain),
        Command::Say {
            common,
            text,
            explain,
        } => say(common, &text, explain),
        Command::Stats { common } => {
            let e = open(common)?;
            let s = e.stats();
            println!(
                "utterances={} learned={} tokens={} vocab={} buf={} topic_k={} meta={} hist={}",
                s.utterances, s.learned, s.tokens, s.vocab, s.buf, s.topic_window, s.meta, s.hist
            );
            Ok(())
        }
        Command::Observe { common, format } => {
            let e = open(common)?;
            let o = e.observe();
            match format.to_ascii_lowercase().as_str() {
                "html" => print!("{}", o.html()),
                _ => print!("{}", o.panel()),
            }
            Ok(())
        }
        Command::Ayumi { common } => {
            let e = open(common)?;
            print!("{}", e.ayumi_text());
            Ok(())
        }
        Command::Rebuild { common } => {
            let mut e = open(common)?;
            let t0 = Instant::now();
            e.rebuild()?;
            println!(
                "rebuilt in {:.3}s  tokens={}",
                t0.elapsed().as_secs_f64(),
                e.stats().tokens
            );
            Ok(())
        }
        Command::Retokenize { common } => {
            let mut e = open(common)?;
            let t0 = Instant::now();
            e.retokenize_from_log()?;
            println!(
                "retokenized in {:.3}s  tokens={} vocab={}",
                t0.elapsed().as_secs_f64(),
                e.stats().tokens,
                e.stats().vocab
            );
            Ok(())
        }
        Command::Bench { tokens } => bench(tokens),
        Command::Verify { sa_tokens, turns } => verify::run(sa_tokens, turns),
        Command::Probe {
            seed,
            triggers,
            rng_seed,
            p_slip,
            p_learn,
        } => {
            let triggers = triggers.or_else(|| {
                let p = PathBuf::from("data/triggers.example.json");
                p.exists().then_some(p)
            });
            probe::run(probe::ProbeArgs {
                seed,
                triggers,
                rng_seed,
                p_slip,
                p_learn,
            })
        }
        Command::Fabricate {
            pairs,
            seed,
            unique_frac,
            out,
        } => {
            let n = munou_engine::fabricate_write(
                &out,
                munou_engine::FabricateOpts {
                    pairs: pairs.max(1),
                    rng_seed: seed,
                    unique_frac,
                },
            )?;
            println!("wrote {n} utterances → {}", out.display());
            Ok(())
        }
        Command::Scale {
            pairs,
            rng_seed,
            unique_frac,
            out,
            p_slip,
            p_learn,
        } => scale::run(scale::ScaleArgs {
            pairs,
            rng_seed,
            unique_frac,
            out,
            p_slip,
            p_learn,
        }),
    }
}

fn params_from(c: &Common) -> Params {
    let mut p = Params::default();
    if let Some(n) = c.n_cand {
        p.n_cand = n.max(1);
    }
    if let Some(s) = c.p_slip {
        p.p_slip = s.clamp(0.0, 1.0);
    }
    if let Some(s) = c.p_learn {
        p.p_learn = s.clamp(0.0, 1.0);
    }
    if let Some(t) = c.tau {
        p.tau_gen = t.max(1e-3);
    }
    if let Some(t) = c.tau_slip {
        p.tau_slip = t.max(1e-3);
    }
    if let Some(b) = c.band_penalty {
        p.band_penalty = b.max(0.0);
    }
    if let Some(s) = c.self_penalty {
        p.self_penalty = s.max(0.0);
    }
    if c.no_bidir {
        p.bidir = false;
    }
    if let Some(n) = c.n_adapt {
        p.n_adapt = n;
    }
    if let Some(w) = c.surprise_weight {
        p.surprise_weight = w;
    }
    if let Some(m) = c.mmr {
        p.mmr_lambda = m.clamp(0.0, 1.0);
    }
    if let Some(n) = c.p_nucleus {
        p.p_nucleus = n.clamp(0.0, 1.0);
    }
    if let Some(k) = c.k_top {
        p.k_top = k;
    }
    if let Some(s) = c.retrieve_scan {
        p.n_retrieve_scan = s;
    }
    if let Some(s) = &c.smoothing {
        p.smoothing = match s.to_ascii_lowercase().as_str() {
            "kn" | "kneser-ney" | "kneserney" | "mkn" | "modified-kn" => SmoothingKind::Kn,
            _ => SmoothingKind::Naive,
        };
    }
    if let Some(x) = c.lambda_skip {
        p.lambda_skip = x.clamp(0.0, 1.0);
    }
    if let Some(x) = c.lambda_cache {
        p.lambda_cache = x.clamp(0.0, 1.0);
    }
    if c.ppm {
        p.ppm_exclude = true;
    }
    if let Some(m) = &c.mix {
        p.mix = match m.to_ascii_lowercase().as_str() {
            "exclusive" | "xor" => MixMode::Exclusive,
            _ => MixMode::Pool,
        };
    }
    if c.no_weather {
        p.weather = false;
    }
    if let Some(r) = c.interject_rate {
        p.interject_rate = r.clamp(0.0, 1.0);
    }
    if let Some(w) = c.interest_weight {
        p.interest_weight = w.max(0.0);
    }
    p
}

fn open(c: Common) -> Result<Engine> {
    let log_path = c.data_dir.join("log.jsonl");
    let cfg = OpenConfig {
        params: params_from(&c),
        seed: c.seed,
        log_path: Some(log_path),
        triggers_path: c.triggers.clone(),
    };
    Engine::open(cfg).context("open engine")
}

fn say(c: Common, text: &str, explain: bool) -> Result<()> {
    let mut e = open(c)?;
    let r = e.respond(text)?;
    if let Some(a) = &r.interject {
        println!("{a}");
    }
    println!("{}", r.text);
    if let Some(m) = &r.milestone {
        println!("（{m}）");
    }
    if explain {
        print!("{}", r.trace.explain_text());
    }
    Ok(())
}

fn chat(c: Common, mut explain: bool) -> Result<()> {
    let mut e = open(c.clone())?;
    let mut spoke = false;
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    writeln!(
        stdout,
        "人工無脳君  seed={}  /observe /why /ayumi /good /bad /stats /eval /rebuild /retok /explain /quit",
        e.seed()
    )?;
    stdout.flush()?;
    let mut line = String::new();
    loop {
        write!(stdout, "> ")?;
        stdout.flush()?;
        line.clear();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        match t {
            "/quit" | "/exit" => break,
            "/observe" => print!("{}", e.observe().panel()),
            "/why" => print!("{}", e.why_text()),
            "/ayumi" => print!("{}", e.ayumi_text()),
            "/good" => writeln!(stdout, "{}", e.feedback(true)?)?,
            "/bad" => writeln!(stdout, "{}", e.feedback(false)?)?,
            "/stats" => {
                let s = e.stats();
                writeln!(
                    stdout,
                    "utterances={} learned={} tokens={} vocab={} buf={} meta={}",
                    s.utterances, s.learned, s.tokens, s.vocab, s.buf, s.meta
                )?;
            }
            "/eval" => writeln!(stdout, "{}", e.eval_summary())?,
            "/rebuild" => {
                e.rebuild()?;
                writeln!(stdout, "ok")?;
            }
            "/retok" => {
                e.retokenize_from_log()?;
                writeln!(stdout, "ok")?;
            }
            "/explain" => {
                explain = !explain;
                writeln!(stdout, "explain={}", explain)?;
            }
            _ => {
                let r = e.respond(t)?;
                spoke = true;
                if let Some(a) = &r.interject {
                    writeln!(stdout, "{a}")?;
                }
                writeln!(stdout, "{}", r.text)?;
                if let Some(m) = &r.milestone {
                    writeln!(stdout, "（{m}）")?;
                }
                writeln!(stdout, "{}", e.observe().strip())?;
                if explain {
                    print!("{}", r.trace.explain_text());
                }
            }
        }
        stdout.flush()?;
    }
    // The session appended turns, so the snapshot cache is stale. Rebuild it
    // now (a fresh open replays and rewrites it) — the cost moves from the
    // next startup to this shutdown.
    if spoke {
        drop(e);
        write!(stdout, "おぼえなおしています…")?;
        stdout.flush()?;
        let t0 = Instant::now();
        let _ = open(c);
        writeln!(stdout, " {:.1}s", t0.elapsed().as_secs_f64())?;
    }
    Ok(())
}

fn bench(n: usize) -> Result<()> {
    let n = n.max(1);
    let mut text = Vec::with_capacity(n);
    let mut x = 0xC0FF_EE00u64;
    for _ in 0..n {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        text.push(((x >> 33) as u32) % 1024 + 16);
    }
    let t0 = Instant::now();
    let sa = munou_engine::suffix_array(&text);
    let dt = t0.elapsed();
    anyhow::ensure!(sa.len() == n, "sa length");
    println!(
        "SA-IS n={}  {:.3}s  {:.1} ns/token",
        n,
        dt.as_secs_f64(),
        dt.as_nanos() as f64 / n as f64
    );
    Ok(())
}
