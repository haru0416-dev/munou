use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use munou_engine::{Engine, OpenConfig, Params, SmoothingKind};

mod verify;

#[derive(Parser, Debug)]
#[command(name = "munou", about = "人工無脳君 — LLM を使わない対話エンジン")]
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
}

#[derive(clap::Args, Debug)]
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
    /// Generation temperature.
    #[arg(long)]
    tau: Option<f64>,
    /// Smoothing: naive | kn
    #[arg(long)]
    smoothing: Option<String>,
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
                "utterances={} tokens={} vocab={} buf={} topic_k={}",
                s.utterances, s.tokens, s.vocab, s.buf, s.topic_window
            );
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
    if let Some(t) = c.tau {
        p.tau_gen = t.max(1e-3);
    }
    if let Some(s) = &c.smoothing {
        p.smoothing = match s.to_ascii_lowercase().as_str() {
            "kn" | "kneser-ney" | "kneserney" => SmoothingKind::Kn,
            _ => SmoothingKind::Naive,
        };
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
    println!("{}", r.text);
    if explain {
        print!("{}", r.trace.explain_text());
    }
    Ok(())
}

fn chat(c: Common, mut explain: bool) -> Result<()> {
    let mut e = open(c)?;
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    writeln!(
        stdout,
        "人工無脳君  seed={}  /why /stats /eval /rebuild /retok /explain /quit",
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
            "/why" => {
                if let Some(tr) = e.last_trace() {
                    print!("{}", tr.explain_text());
                } else {
                    writeln!(stdout, "(no trace yet)")?;
                }
            }
            "/stats" => {
                let s = e.stats();
                writeln!(
                    stdout,
                    "utterances={} tokens={} vocab={} buf={}",
                    s.utterances, s.tokens, s.vocab, s.buf
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
                writeln!(stdout, "{}", r.text)?;
                if explain {
                    print!("{}", r.trace.explain_text());
                }
            }
        }
        stdout.flush()?;
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
