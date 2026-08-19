use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rustc_hash::FxHashSet;

use crate::embed::{cosine, Embedder, HashEmbedder, TopicTracker};
use crate::error::Result;
use crate::eval::EvalAccum;
use crate::explain::{PathKind, Trace};
use crate::generate::{generate_one, lcs_len};
use crate::ids::{is_special, TokenId};
use crate::intern::Interner;
use crate::log::{now_ms, AppendLog, Record, Role};
use crate::mix::Pool;
use crate::observe::Observe;
use crate::params::{MixMode, Params};
use crate::select::{rank_and_pick, RankInput};
use crate::smoothing::{self, Smoothing};
use crate::store::Store;
use crate::tokenizer::{detokenize, Tokenized, Tokenizer};
use crate::trigger::TriggerDict;

#[derive(Default)]
pub struct OpenConfig {
    pub params: Params,
    pub seed: u64,
    pub log_path: Option<PathBuf>,
    pub triggers_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct Reply {
    pub text: String,
    pub trace: Trace,
}

#[derive(Debug, Clone)]
pub struct Stats {
    pub utterances: usize,
    pub learned: usize,
    pub tokens: usize,
    pub vocab: usize,
    pub buf: usize,
    pub topic_window: usize,
}

pub struct Engine {
    intern: Interner,
    tokenizer: Tokenizer,
    store: Store,
    embedder: HashEmbedder,
    topic: TopicTracker,
    triggers: TriggerDict,
    params: Params,
    rng: ChaCha8Rng,
    seed: u64,
    log: AppendLog,
    last_trace: Option<Trace>,
    smoothing: Box<dyn Smoothing>,
    eval: EvalAccum,
    history: VecDeque<TokenId>,
    prior: Vec<Vec<TokenId>>,
    /// Past bot utterances for retrieval (text + chunk ids).
    bots: Vec<(String, Vec<TokenId>)>,
}

impl Engine {
    pub fn ephemeral(params: Params, seed: u64) -> Result<Self> {
        Self::open(OpenConfig {
            params,
            seed,
            log_path: None,
            triggers_path: None,
        })
    }

    pub fn open(cfg: OpenConfig) -> Result<Self> {
        let embedder = HashEmbedder::from_params(&cfg.params);
        let mut tokenizer = Tokenizer::new(&cfg.params);
        let log = AppendLog::open(cfg.log_path.as_deref())?;
        for rec in &log.records {
            if rec.learned {
                tokenizer.observe(&rec.text);
            }
        }

        let triggers = if let Some(p) = cfg.triggers_path.as_deref() {
            if p.exists() {
                TriggerDict::from_path(p)?
            } else {
                TriggerDict::default()
            }
        } else {
            TriggerDict::default()
        };

        let mut intern = Interner::new();
        let mut store = Store::new(cfg.params.merge_threshold);
        let mut topic = TopicTracker::new(cfg.params.embed_dim, cfg.params.k_topic);
        let mut history = VecDeque::new();
        let mut prior = Vec::new();
        let mut bots = Vec::new();
        let dim = cfg.params.embed_dim;

        for rec in &log.records {
            let tok = tokenizer.tokenize(&mut intern, &rec.text);
            if rec.learned {
                store.push_utterance(&tok.chunks);
                prior.push(tok.chunks.clone());
                if rec.role == Role::Bot {
                    bots.push((rec.text.clone(), tok.chunks.clone()));
                }
            }
            for &c in &tok.chunks {
                history.push_back(c);
            }
            let mut v = vec![0.0f32; dim];
            embedder.embed(&rec.text, &mut v);
            topic.push(&v);
        }
        store.merge();
        trim_history(&mut history, cfg.params.l_max_capped() * 4);

        let mut eval = EvalAccum::default();
        for rec in &log.records {
            eval.ingest_bot(rec, &cfg.params);
        }

        let smoothing = smoothing::boxed(cfg.params.smoothing, cfg.params.kn_discount);
        Ok(Self {
            intern,
            tokenizer,
            store,
            embedder,
            topic,
            triggers,
            rng: ChaCha8Rng::seed_from_u64(cfg.seed),
            seed: cfg.seed,
            params: cfg.params,
            log,
            last_trace: None,
            smoothing,
            eval,
            history,
            prior,
            bots,
        })
    }

    pub fn load_triggers(&mut self, path: &Path) -> Result<()> {
        self.triggers = TriggerDict::from_path(path)?;
        Ok(())
    }

    pub fn respond(&mut self, input: &str) -> Result<Reply> {
        let t0 = Instant::now();
        let input = input.trim();
        let tok = self.tokenizer.tokenize(&mut self.intern, input);

        let mut q = vec![0.0f32; self.embedder.dim()];
        self.embedder.embed(input, &mut q);
        self.topic.push(&q);
        let mut topic = vec![0.0f32; self.embedder.dim()];
        self.topic.mean(&mut topic);

        let trigger_hit = self.triggers.match_one(&self.embedder, input, &self.params);

        let mut pool = Pool::default();
        let mut steps = Vec::new();
        let mut trigger_tr = None;

        if let Some((tr, responses)) = trigger_hit {
            trigger_tr = Some(tr);
            for r in responses {
                let tk = self.tokenizer.tokenize(&mut self.intern, &r);
                pool.push(PathKind::Trigger, r, tk.chunks);
            }
        }

        let exclusive_skip_rest = self.params.mix == MixMode::Exclusive && !pool.is_empty();

        if !exclusive_skip_rest {
            if self.params.mix == MixMode::Pool {
                self.propose_retrieve(&mut pool, input, &topic);
            }
            let markov_ok = self.params.mix == MixMode::Exclusive || !self.store.is_empty();
            if markov_ok {
                self.propose_markov(&mut pool, &tok.chunks, &mut steps);
            }
            if self.params.mix == MixMode::Pool {
                self.propose_echo(&mut pool, &tok);
            }
        }

        if pool.is_empty() {
            self.propose_echo(&mut pool, &tok);
        }

        let texts = pool.texts();
        let toks = pool.tokens();
        let sources = pool.sources();

        let ranked = rank_and_pick(
            &self.embedder,
            RankInput {
                topic: &topic,
                texts: &texts,
                tokens: &toks,
                sources: &sources,
                input_tokens: &tok.chunks,
                trigger_match: trigger_tr.as_ref().map(|t| t.similarity).unwrap_or(0.0),
            },
            &self.params,
            &mut self.rng,
        );

        let path = sources
            .get(ranked.index)
            .copied()
            .unwrap_or(PathKind::Markov);
        let chosen_tokens = toks.get(ranked.index).cloned().unwrap_or_default();
        let chosen_text = texts
            .get(ranked.index)
            .cloned()
            .unwrap_or_else(|| "…".into());
        let sim = ranked
            .traces
            .iter()
            .find(|c| c.chosen)
            .map(|c| c.topic_score)
            .unwrap_or_else(|| {
                let mut b = vec![0.0; self.embedder.dim()];
                self.embedder.embed(&chosen_text, &mut b);
                cosine(&topic, &b)
            });
        let band_hit = sim >= self.params.band_lo && sim <= self.params.band_hi;
        let novelty_lcs = self
            .prior
            .iter()
            .map(|u| lcs_len(&chosen_tokens, u))
            .max()
            .unwrap_or(0);
        let chosen_rank = ranked
            .traces
            .iter()
            .find(|c| c.chosen)
            .map(|c| c.rank)
            .unwrap_or(0);

        let learn_roll: f64 = self.rng.gen();
        let learned = learn_roll < self.params.p_learn.clamp(0.0, 1.0);

        let trace = Trace {
            seed: self.seed,
            path,
            input: input.to_string(),
            morphemes: tok
                .morphemes
                .iter()
                .map(|id| self.intern.get(*id).to_string())
                .collect(),
            chunks: tok
                .chunks
                .iter()
                .map(|id| self.intern.get(*id).to_string())
                .collect(),
            topic_hits: self.topic.len(),
            trigger: trigger_tr,
            candidates: ranked.traces,
            chosen_rank,
            slipped: ranked.slipped,
            slip_roll: ranked.slip_roll,
            p_slip: self.params.p_slip,
            learned,
            learn_roll,
            p_learn: self.params.p_learn,
            steps,
            elapsed_us: t0.elapsed().as_micros(),
            novelty_lcs,
            similarity: sim,
            band_hit,
        };

        self.eval.observe(&trace, chosen_tokens.len());
        self.log.append(Record {
            v: 1,
            t: now_ms(),
            role: Role::User,
            text: input.to_string(),
            slipped: None,
            score: None,
            learned,
            path: None,
            novelty_lcs: None,
            n_tok: None,
        })?;
        self.log.append(Record {
            v: 1,
            t: now_ms(),
            role: Role::Bot,
            text: chosen_text.clone(),
            slipped: Some(ranked.slipped),
            score: Some(sim),
            learned,
            path: Some(path),
            novelty_lcs: Some(novelty_lcs),
            n_tok: Some(chosen_tokens.len()),
        })?;
        if learned {
            self.absorb(Role::User, input, &tok.chunks);
            self.absorb(Role::Bot, &chosen_text, &chosen_tokens);
        }
        for &c in tok.chunks.iter().chain(chosen_tokens.iter()) {
            self.history.push_back(c);
        }
        trim_history(&mut self.history, self.params.l_max_capped() * 4);
        self.last_trace = Some(trace.clone());

        Ok(Reply {
            text: chosen_text,
            trace,
        })
    }

    fn absorb(&mut self, role: Role, text: &str, chunks: &[TokenId]) {
        self.tokenizer.observe(text);
        if chunks.is_empty() {
            return;
        }
        self.store.push_utterance(chunks);
        self.prior.push(chunks.to_vec());
        if role == Role::Bot {
            self.bots.push((text.to_string(), chunks.to_vec()));
        }
    }

    fn propose_markov(
        &mut self,
        pool: &mut Pool,
        user_chunks: &[TokenId],
        steps: &mut Vec<crate::explain::GenStep>,
    ) {
        let mut ctx_seed: Vec<TokenId> = self.history.iter().copied().collect();
        ctx_seed.extend_from_slice(user_chunks);
        let parrot: Vec<TokenId> = if self.params.mix == MixMode::Pool && !self.store.is_empty() {
            Vec::new()
        } else {
            user_chunks.to_vec()
        };
        let mut seen = FxHashSet::default();
        let mut attempts = 0;
        let start = pool.items.len();
        while pool.items.len() - start < self.params.n_cand && attempts < self.params.n_cand * 4 {
            attempts += 1;
            let g = generate_one(
                &self.store,
                self.smoothing.as_ref(),
                &self.params,
                &ctx_seed,
                &parrot,
                &mut self.rng,
            );
            if g.tokens.is_empty() {
                continue;
            }
            if !seen.insert(g.tokens.clone()) {
                continue;
            }
            let chunk_strs: Vec<String> = g
                .tokens
                .iter()
                .copied()
                .filter(|id| !is_special(*id))
                .map(|id| self.intern.get(id).to_string())
                .collect();
            let text = detokenize(&chunk_strs);
            if text.trim().is_empty() {
                continue;
            }
            if steps.is_empty() {
                *steps = g.steps;
            }
            pool.push(PathKind::Markov, text, g.tokens);
        }
        if self.params.mix == MixMode::Exclusive && pool.is_empty() {
            let parrot_strs: Vec<String> = user_chunks
                .iter()
                .map(|id| self.intern.get(*id).to_string())
                .collect();
            let parrot_text = detokenize(&parrot_strs);
            let shuffled = parrot_variant(&parrot_strs, &mut self.rng);
            pool.push(PathKind::Markov, shuffled, user_chunks.to_vec());
            if parrot_text != pool.items.last().map(|p| p.text.as_str()).unwrap_or("") {
                pool.push(PathKind::Markov, parrot_text, user_chunks.to_vec());
            }
        }
    }

    fn propose_retrieve(&mut self, pool: &mut Pool, input: &str, topic: &[f32]) {
        if self.bots.is_empty() || self.params.n_retrieve == 0 {
            return;
        }
        let mut buf = vec![0.0f32; self.embedder.dim()];
        let mut scored: Vec<(f32, usize)> = Vec::with_capacity(self.bots.len());
        for (i, (text, _)) in self.bots.iter().enumerate() {
            if text == input {
                continue;
            }
            self.embedder.embed(text, &mut buf);
            scored.push((cosine(topic, &buf), i));
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut n = 0usize;
        for (_, i) in scored {
            if n >= self.params.n_retrieve {
                break;
            }
            let (text, toks) = self.bots[i].clone();
            let before = pool.items.len();
            pool.push(PathKind::Retrieve, text, toks);
            if pool.items.len() > before {
                n += 1;
            }
        }
    }

    fn propose_echo(&mut self, pool: &mut Pool, tok: &Tokenized) {
        let parrot_strs: Vec<String> = tok
            .chunks
            .iter()
            .map(|id| self.intern.get(*id).to_string())
            .collect();
        let parrot_text = detokenize(&parrot_strs);
        pool.push(PathKind::Echo, parrot_text, tok.chunks.clone());
        if self.params.n_echo >= 2 {
            let shuffled = parrot_variant(&parrot_strs, &mut self.rng);
            pool.push(PathKind::Echo, shuffled, tok.chunks.clone());
        }
    }

    pub fn rebuild(&mut self) -> Result<()> {
        self.store.merge();
        Ok(())
    }

    pub fn last_trace(&self) -> Option<&Trace> {
        self.last_trace.as_ref()
    }

    pub fn stats(&self) -> Stats {
        Stats {
            utterances: self.log.records.len(),
            learned: self.log.records.iter().filter(|r| r.learned).count(),
            tokens: self.store.len(),
            vocab: self.intern.vocab_user(),
            buf: self.store.buf.len(),
            topic_window: self.topic.len(),
        }
    }

    pub fn eval_summary(&self) -> String {
        self.eval.summary(&self.params)
    }

    pub fn observe(&self) -> Observe {
        Observe::from_parts(
            &self.stats(),
            &self.params,
            &self.log.records,
            self.last_trace.as_ref(),
            &self.eval,
        )
    }

    pub fn params(&self) -> &Params {
        &self.params
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// One Markov draw without embedding, logging, or topic update.
    /// Used by `munou verify` to isolate engine-path latency from hash-embed.
    pub fn markov_draw(&mut self) -> usize {
        let ctx: Vec<TokenId> = self.history.iter().copied().collect();
        let parrot: Vec<TokenId> = ctx.iter().rev().copied().take(4).collect();
        generate_one(
            &self.store,
            self.smoothing.as_ref(),
            &self.params,
            &ctx,
            &parrot,
            &mut self.rng,
        )
        .tokens
        .len()
    }

    /// Rebuild tokenizer + SA from the log (source of truth).
    pub fn retokenize_from_log(&mut self) -> Result<()> {
        let recs: Vec<(Role, String, bool)> = self
            .log
            .records
            .iter()
            .map(|r| (r.role, r.text.clone(), r.learned))
            .collect();
        self.tokenizer = Tokenizer::new(&self.params);
        for (_, t, learned) in &recs {
            if *learned {
                self.tokenizer.observe(t);
            }
        }
        self.intern = Interner::new();
        self.store = Store::new(self.params.merge_threshold);
        self.history.clear();
        self.prior.clear();
        self.bots.clear();
        self.topic = TopicTracker::new(self.embedder.dim(), self.params.k_topic);
        for (role, t, learned) in &recs {
            let tok = self.tokenizer.tokenize(&mut self.intern, t);
            if *learned {
                self.store.push_utterance(&tok.chunks);
                self.prior.push(tok.chunks.clone());
                if *role == Role::Bot {
                    self.bots.push((t.clone(), tok.chunks.clone()));
                }
            }
            for &c in &tok.chunks {
                self.history.push_back(c);
            }
            let mut v = vec![0.0f32; self.embedder.dim()];
            self.embedder.embed(t, &mut v);
            self.topic.push(&v);
        }
        self.store.merge();
        trim_history(&mut self.history, self.params.l_max_capped() * 4);
        Ok(())
    }
}

fn trim_history(h: &mut VecDeque<TokenId>, cap: usize) {
    while h.len() > cap {
        h.pop_front();
    }
}

fn parrot_variant<R: rand::Rng + ?Sized>(chunks: &[String], rng: &mut R) -> String {
    if chunks.is_empty() {
        return "…".into();
    }
    let mut v = chunks.to_vec();
    // Fisher–Yates, but keep it mild: swap a couple of neighbours
    if v.len() > 1 {
        let i = rng.gen_range(0..v.len() - 1);
        v.swap(i, i + 1);
    }
    detokenize(&v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_given_seed() {
        let lines = ["こんにちは", "今日はいい天気だね", "そうだね", "また今度"];
        let mut a = Engine::ephemeral(Params::default(), 42).unwrap();
        let mut b = Engine::ephemeral(Params::default(), 42).unwrap();
        for line in lines {
            let ra = a.respond(line).unwrap();
            let rb = b.respond(line).unwrap();
            assert_eq!(ra.text, rb.text, "line={line}");
        }
    }

    #[test]
    fn different_seeds_can_diverge() {
        let mut a = Engine::ephemeral(Params::default(), 1).unwrap();
        let mut b = Engine::ephemeral(Params::default(), 2).unwrap();
        let mut same = 0;
        let mut n = 0;
        for line in ["こんにちは", "おはよう", "ねむい", "ごはん", "また"] {
            let ra = a.respond(line).unwrap();
            let rb = b.respond(line).unwrap();
            n += 1;
            if ra.text == rb.text {
                same += 1;
            }
        }
        assert!(same < n || n == 0);
    }

    #[test]
    fn log_survives_reopen() {
        let dir = std::env::temp_dir().join(format!(
            "munou-persist-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let log = dir.join("log.jsonl");
        {
            let mut e = Engine::open(OpenConfig {
                params: Params::default(),
                seed: 9,
                log_path: Some(log.clone()),
                triggers_path: None,
            })
            .unwrap();
            e.respond("こんにちは").unwrap();
        }
        let e2 = Engine::open(OpenConfig {
            params: Params::default(),
            seed: 9,
            log_path: Some(log.clone()),
            triggers_path: None,
        })
        .unwrap();
        assert!(e2.stats().utterances >= 2);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
