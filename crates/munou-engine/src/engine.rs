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
use crate::generate::{generate_one, NextMemo};
use crate::ids::{is_special, TokenId};
use crate::intern::Interner;
use crate::log::{now_ms, AppendLog, Record, Role};
use crate::mix::Pool;
use crate::observe::Observe;
use crate::params::{MixMode, Params, SmoothingKind};
use crate::route;
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
    pub episodic: usize,
    pub meta: usize,
    pub hist: usize,
    pub path_prior: [f32; 4],
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
    /// Past bot utterances for retrieval (text + chunk ids + cached embedding).
    /// Trimmed to the retrieve scan window when `n_retrieve_scan > 0`.
    bots: Vec<BotUtterance>,
    /// Closed analog of RLHF: additive path prior from `/good` `/bad`.
    path_prior: [f32; 4],
    /// Last bot path, restored from the log so `/good` works after reopen.
    last_path: Option<PathKind>,
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
            if rec.role == Role::Meta {
                continue;
            }
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
        let mut bots: Vec<BotUtterance> = Vec::new();
        let dim = cfg.params.embed_dim;

        let mut path_prior = [0.0f32; 4];
        let mut last_bot_path = None;
        for rec in &log.records {
            match rec.role {
                Role::Meta => {
                    let pk = rec.path.or(last_bot_path);
                    if let Some(pk) = pk {
                        apply_pref(
                            &mut path_prior,
                            pk,
                            rec.text == "good",
                            cfg.params.pref_step,
                            cfg.params.pref_clip,
                        );
                    }
                }
                Role::Bot => last_bot_path = rec.path,
                Role::User => {}
            }
        }

        // Only the last k records can still be inside the topic window, so
        // embedding every record on open is wasted work — the window state is
        // identical either way.
        let n_speech = log.records.iter().filter(|r| r.role != Role::Meta).count();
        let k_topic = cfg.params.k_topic.max(1);
        let mut speech_idx = 0usize;
        for rec in &log.records {
            if rec.role == Role::Meta {
                continue;
            }
            let tok = tokenizer.tokenize(&mut intern, &rec.text);
            if rec.learned {
                store.push_utterance(&tok.chunks);
                if rec.role == Role::Bot {
                    bots.push(BotUtterance {
                        text: rec.text.clone(),
                        toks: tok.chunks.clone(),
                        emb: Vec::new(),
                    });
                }
            }
            for &c in &tok.chunks {
                history.push_back(c);
            }
            if speech_idx + k_topic >= n_speech {
                let mut v = vec![0.0f32; dim];
                embedder.embed(&rec.text, &mut v);
                topic.push(&v);
            }
            speech_idx += 1;
        }
        store.merge();
        trim_history(&mut history, cfg.params.l_max_capped() * 4);
        finish_bots(&mut bots, &embedder, cfg.params.n_retrieve_scan);

        let mut eval = EvalAccum::default();
        for rec in &log.records {
            eval.ingest_bot(rec, &cfg.params);
        }

        let mut smoothing = smoothing::boxed(cfg.params.smoothing, cfg.params.kn_discount);
        smoothing::sync_to_store(smoothing.as_mut(), &cfg.params, &store);
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
            bots,
            path_prior,
            last_path: last_bot_path,
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
        let retr_sim = self.max_bot_sim(&topic);
        let trig_sim = trigger_hit
            .as_ref()
            .map(|(t, _)| t.similarity)
            .unwrap_or(0.0);
        let route = route::plan(
            &self.params,
            self.store.len(),
            self.bots.len(),
            retr_sim,
            trig_sim,
        );

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
            if route.run_retrieve {
                self.propose_retrieve(&mut pool, input, &topic, route.n_retrieve);
            }
            let markov_ok = route.run_markov
                && (self.params.mix == MixMode::Exclusive || !self.store.is_empty());
            let n_cand = if self.params.mix == MixMode::Exclusive {
                route.n_cand.max(1)
            } else {
                route.n_cand
            };
            if markov_ok && n_cand > 0 {
                self.propose_markov(&mut pool, &tok.chunks, &mut steps, n_cand);
            }
            if route.run_echo {
                self.propose_echo(&mut pool, &tok);
            }
        }

        if pool.is_empty() {
            self.propose_echo(&mut pool, &tok);
        }
        if pool.is_empty() {
            pool.push(PathKind::Echo, "…".into(), Vec::new());
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
                path_prior: self.path_prior,
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
        let novelty_lcs = longest_corpus_run(&self.store, &chosen_tokens);
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
            route: Some(route.explain_line()),
            path_prior: self.path_prior,
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
        self.last_path = Some(path);

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
        if role == Role::Bot {
            let mut emb = vec![0.0f32; self.embedder.dim()];
            self.embedder.embed(text, &mut emb);
            self.bots.push(BotUtterance {
                text: text.to_string(),
                toks: chunks.to_vec(),
                emb,
            });
            trim_bots(&mut self.bots, self.params.n_retrieve_scan);
        }
        smoothing::sync_to_store(self.smoothing.as_mut(), &self.params, &self.store);
    }

    fn propose_markov(
        &mut self,
        pool: &mut Pool,
        user_chunks: &[TokenId],
        steps: &mut Vec<crate::explain::GenStep>,
        n_cand: usize,
    ) {
        let mut ctx_seed: Vec<TokenId> = self.history.iter().copied().collect();
        ctx_seed.extend_from_slice(user_chunks);
        let parrot: Vec<TokenId> = if self.params.mix == MixMode::Pool && !self.store.is_empty() {
            Vec::new()
        } else {
            user_chunks.to_vec()
        };
        let kn = matches!(self.params.smoothing, SmoothingKind::Kn);
        self.store.warm_sampling(kn);
        let mut memo = NextMemo::default();
        let mut seen = FxHashSet::default();
        let mut attempts = 0;
        let start = pool.items.len();
        while pool.items.len() - start < n_cand && attempts < n_cand.max(1) * 4 {
            attempts += 1;
            let uni = self.store.sampling_view(kn).expect("warmed above");
            let g = generate_one(
                &self.store,
                self.smoothing.as_ref(),
                &self.params,
                &ctx_seed,
                &parrot,
                uni,
                &mut memo,
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

    fn bot_scan_start(&self) -> usize {
        let n = self.bots.len();
        let cap = self.params.n_retrieve_scan;
        if cap == 0 || cap >= n {
            0
        } else {
            n - cap
        }
    }

    fn propose_retrieve(&self, pool: &mut Pool, input: &str, topic: &[f32], n_retrieve: usize) {
        if self.bots.is_empty() || n_retrieve == 0 {
            return;
        }
        let lambda = self.params.mmr_lambda.clamp(0.0, 1.0);
        let start = self.bot_scan_start();
        let mut cands: Vec<(f32, usize)> = Vec::with_capacity(self.bots.len() - start);
        for (i, b) in self.bots.iter().enumerate().skip(start) {
            if b.text == input {
                continue;
            }
            cands.push((cosine(topic, &b.emb), i));
        }
        // MMR with an incrementally maintained max-redundancy per candidate:
        // red_i = max over picked j of cos(e_i, e_j) — same value as the old
        // per-round fold, updated in O(n) per pick instead of recomputed in
        // O(n·k) cosines per round.
        let mut picked_flag = vec![false; cands.len()];
        let mut red = vec![0.0f32; cands.len()];
        let mut n_picked = 0usize;
        while n_picked < n_retrieve && n_picked < cands.len() {
            let mut best_i = None;
            let mut best_s = f32::NEG_INFINITY;
            for (ci, &(sim, bot_i)) in cands.iter().enumerate() {
                if picked_flag[ci] {
                    continue;
                }
                if pool.items.iter().any(|p| p.text == self.bots[bot_i].text) {
                    continue;
                }
                let mmr = lambda * sim - (1.0 - lambda) * red[ci];
                if mmr > best_s {
                    best_s = mmr;
                    best_i = Some(ci);
                }
            }
            let Some(ci) = best_i else {
                break;
            };
            picked_flag[ci] = true;
            n_picked += 1;
            let bot_i = cands[ci].1;
            let b = &self.bots[bot_i];
            pool.push(PathKind::Retrieve, b.text.clone(), b.toks.clone());
            let picked_emb = &self.bots[bot_i].emb;
            for (cj, &(_, bj)) in cands.iter().enumerate() {
                if !picked_flag[cj] {
                    let r = cosine(&self.bots[bj].emb, picked_emb);
                    if r > red[cj] {
                        red[cj] = r;
                    }
                }
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
        smoothing::sync_to_store(self.smoothing.as_mut(), &self.params, &self.store);
        Ok(())
    }

    pub fn last_trace(&self) -> Option<&Trace> {
        self.last_trace.as_ref()
    }

    /// `/why` surface. After reopen the gen chain is gone; we still show the last bot line.
    pub fn why_text(&self) -> String {
        if let Some(tr) = &self.last_trace {
            return tr.explain_text();
        }
        match self.observe().last_why {
            Some(why) => {
                format!("{why}\n(reopened; gen chain lives only in the process that spoke)\n")
            }
            None => "(no trace yet)\n".into(),
        }
    }

    /// Closed analog of a preference label. Does not call a reward model.
    pub fn feedback(&mut self, good: bool) -> Result<String> {
        let path = self.last_trace.as_ref().map(|t| t.path).or(self.last_path);
        let Some(path) = path else {
            return Ok("no turn yet".into());
        };
        apply_pref(
            &mut self.path_prior,
            path,
            good,
            self.params.pref_step,
            self.params.pref_clip,
        );
        self.log.append(Record {
            v: 1,
            t: now_ms(),
            role: Role::Meta,
            text: if good { "good".into() } else { "bad".into() },
            slipped: None,
            score: None,
            learned: false,
            path: Some(path),
            novelty_lcs: None,
            n_tok: None,
        })?;
        Ok(format!(
            "pref {} {:?}  prior={:+.2}",
            if good { "good" } else { "bad" },
            path,
            self.path_prior[route::prior_index(path)]
        ))
    }

    fn max_bot_sim(&self, topic: &[f32]) -> f32 {
        if self.bots.is_empty() {
            return 0.0;
        }
        let mut m = 0.0f32;
        for b in self.bots.iter().skip(self.bot_scan_start()) {
            m = m.max(cosine(topic, &b.emb));
        }
        m
    }

    pub fn stats(&self) -> Stats {
        let episodic = self
            .log
            .records
            .iter()
            .filter(|r| r.role != Role::Meta)
            .count();
        let meta = self
            .log
            .records
            .iter()
            .filter(|r| r.role == Role::Meta)
            .count();
        let learned = self
            .log
            .records
            .iter()
            .filter(|r| r.role != Role::Meta && r.learned)
            .count();
        Stats {
            utterances: episodic,
            learned,
            tokens: self.store.len(),
            vocab: self.intern.vocab_user(),
            buf: self.store.buf.len(),
            topic_window: self.topic.len(),
            episodic,
            meta,
            hist: self.history.len(),
            path_prior: self.path_prior,
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
        let kn = matches!(self.params.smoothing, SmoothingKind::Kn);
        self.store.warm_sampling(kn);
        let uni = self.store.sampling_view(kn).expect("warmed above");
        let mut memo = NextMemo::default();
        generate_one(
            &self.store,
            self.smoothing.as_ref(),
            &self.params,
            &ctx,
            &parrot,
            uni,
            &mut memo,
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
        for (role, t, learned) in &recs {
            if *role == Role::Meta || !*learned {
                continue;
            }
            self.tokenizer.observe(t);
        }
        self.intern = Interner::new();
        self.store = Store::new(self.params.merge_threshold);
        self.history.clear();
        self.bots.clear();
        self.topic = TopicTracker::new(self.embedder.dim(), self.params.k_topic);
        let n_speech = recs
            .iter()
            .filter(|(role, _, _)| *role != Role::Meta)
            .count();
        let k_topic = self.params.k_topic.max(1);
        let mut speech_idx = 0usize;
        for (role, t, learned) in &recs {
            if *role == Role::Meta {
                continue;
            }
            let tok = self.tokenizer.tokenize(&mut self.intern, t);
            if *learned {
                self.store.push_utterance(&tok.chunks);
                if *role == Role::Bot {
                    self.bots.push(BotUtterance {
                        text: t.clone(),
                        toks: tok.chunks.clone(),
                        emb: Vec::new(),
                    });
                }
            }
            for &c in &tok.chunks {
                self.history.push_back(c);
            }
            if speech_idx + k_topic >= n_speech {
                let mut v = vec![0.0f32; self.embedder.dim()];
                self.embedder.embed(t, &mut v);
                self.topic.push(&v);
            }
            speech_idx += 1;
        }
        self.store.merge();
        trim_history(&mut self.history, self.params.l_max_capped() * 4);
        finish_bots(&mut self.bots, &self.embedder, self.params.n_retrieve_scan);
        smoothing::sync_to_store(self.smoothing.as_mut(), &self.params, &self.store);
        Ok(())
    }
}

fn apply_pref(prior: &mut [f32; 4], path: PathKind, good: bool, step: f32, clip: f32) {
    let i = route::prior_index(path);
    let d = if good { step } else { -step };
    let clip = clip.max(0.0);
    prior[i] = (prior[i] + d).clamp(-clip, clip);
}

#[derive(Debug, Clone)]
struct BotUtterance {
    text: String,
    toks: Vec<TokenId>,
    /// Hash embedding of `text`, cached so retrieve / routing do not re-embed
    /// the scan window every turn. Same embedder, same values.
    emb: Vec<f32>,
}

/// Drop bots that can never enter the scan window again, then fill embeddings.
/// Called once at the end of open / retokenize.
fn finish_bots(bots: &mut Vec<BotUtterance>, embedder: &HashEmbedder, cap: usize) {
    if cap > 0 && bots.len() > cap {
        let cut = bots.len() - cap;
        bots.drain(..cut);
    }
    for b in bots.iter_mut() {
        let mut v = vec![0.0f32; embedder.dim()];
        embedder.embed(&b.text, &mut v);
        b.emb = v;
    }
}

/// Amortised front-trim: entries before the last `cap` are unreachable
/// (`bot_scan_start`), so keeping at most 2·cap preserves behaviour exactly.
fn trim_bots(bots: &mut Vec<BotUtterance>, cap: usize) {
    if cap > 0 && bots.len() > cap * 2 {
        let cut = bots.len() - cap;
        bots.drain(..cut);
    }
}

/// Longest run of `toks` occurring anywhere in the corpus. Utterances are
/// wrapped SEP..EOS in the store and chunk runs never contain specials, so an
/// occurrence cannot cross an utterance boundary — this equals the previous
/// max-over-all-prior-utterances `lcsubstr_len`, computed against the suffix
/// array instead of re-scanning every utterance per turn.
fn longest_corpus_run(store: &Store, toks: &[TokenId]) -> usize {
    let mut best = 0usize;
    for seg in toks.split(|t| is_special(*t)) {
        let m = seg.len();
        for i in 0..m {
            if i + best >= m {
                break;
            }
            let mut l = best + 1;
            while i + l <= m && store.contains_seq(&seg[i..i + l]) {
                l += 1;
            }
            best = best.max(l - 1);
        }
    }
    best
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

    #[test]
    fn feedback_is_meta_not_corpus_and_survives_reopen() {
        let dir = std::env::temp_dir().join(format!(
            "munou-pref-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let log = dir.join("log.jsonl");
        let params = Params {
            p_learn: 1.0,
            p_slip: 0.0,
            ..Params::default()
        };
        let prior_after;
        let tokens_after;
        let utterances_after;
        let path;
        {
            let mut e = Engine::open(OpenConfig {
                params: params.clone(),
                seed: 9,
                log_path: Some(log.clone()),
                triggers_path: None,
            })
            .unwrap();
            let r = e.respond("こんにちは").unwrap();
            path = r.trace.path;
            tokens_after = e.stats().tokens;
            utterances_after = e.stats().utterances;
            let msg = e.feedback(true).unwrap();
            assert!(msg.contains("good"), "{msg}");
            assert_eq!(e.stats().tokens, tokens_after);
            assert_eq!(e.stats().utterances, utterances_after);
            assert_eq!(e.stats().meta, 1);
            assert!(e.stats().path_prior[route::prior_index(path)] > 0.0);
            prior_after = e.stats().path_prior;
            assert!(e.observe().panel().contains("好み"));
        }
        let e2 = Engine::open(OpenConfig {
            params,
            seed: 9,
            log_path: Some(log.clone()),
            triggers_path: None,
        })
        .unwrap();
        assert_eq!(e2.stats().tokens, tokens_after);
        assert_eq!(e2.stats().utterances, utterances_after);
        assert_eq!(e2.stats().learned, utterances_after);
        assert_eq!(e2.stats().meta, 1);
        assert_eq!(e2.stats().path_prior, prior_after);
        assert!(e2.stats().path_prior[route::prior_index(path)] > 0.0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
