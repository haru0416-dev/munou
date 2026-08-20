use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rustc_hash::FxHashSet;

use crate::adapt::PairStore;
use crate::embed::{cosine, Embedder, HashEmbedder, TopicTracker};
use crate::error::Result;
use crate::eval::EvalAccum;
use crate::explain::{GenStep, PathKind, Trace};
use crate::generate::{generate_one, Generated, NextMemo};
use crate::ids::{is_special, TokenId};
use crate::intern::Interner;
use crate::log::{AppendLog, Record, Role};
use crate::mix::Pool;
use crate::observe::Observe;
use crate::params::{MixMode, Params, SmoothingKind};
use crate::retrieve::BotStore;
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
    pub path_prior: [f32; 5],
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
    /// Retrieve source: past bot utterances with cached embeddings.
    bots: BotStore,
    /// Adapt source: learned (user → reply) exchanges, Reudy analog.
    pairs: PairStore,
    /// Reversed-stream twin of `store` for keyword-anchored bidirectional
    /// generation (MegaHAL analog). Empty when `bidir` is off.
    rev_store: Store,
    /// Closed analog of RLHF: additive path prior from `/good` `/bad`.
    path_prior: [f32; 5],
    /// Last bot path, restored from the log so `/good` works after reopen.
    last_path: Option<PathKind>,
    /// Last `self_window` own reply texts for the self-repetition penalty.
    /// Text, not tokens: the penalty is char-level so live and replay agree
    /// exactly (token ids drift across reopen).
    recent_bot: VecDeque<String>,
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

        let mut triggers = if let Some(p) = cfg.triggers_path.as_deref() {
            if p.exists() {
                TriggerDict::from_path(p)?
            } else {
                TriggerDict::default()
            }
        } else {
            TriggerDict::default()
        };
        triggers.warm(&embedder);

        let mut path_prior = [0.0f32; 5];
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

        let recs: Vec<(Role, &str, bool)> = log
            .records
            .iter()
            .map(|r| (r.role, r.text.as_str(), r.learned))
            .collect();
        let replayed = replay_speech(&cfg.params, &embedder, &mut tokenizer, &recs);
        let Replayed {
            intern,
            store,
            topic,
            history,
            bots,
            recent_bot,
            pairs,
            rev_store,
        } = replayed;

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
            pairs,
            rev_store,
            path_prior,
            last_path: last_bot_path,
            recent_bot,
        })
    }

    pub fn load_triggers(&mut self, path: &Path) -> Result<()> {
        self.triggers = TriggerDict::from_path(path)?;
        self.triggers.warm(&self.embedder);
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

        let (pool, steps, trigger_tr, route) = self.propose_all(input, &tok, &topic, &q);

        let texts = pool.texts();
        let toks = pool.tokens();
        let sources = pool.sources();
        let surprises = pool.surprises();

        self.recent_bot.make_contiguous();
        let (recent_bot, _) = self.recent_bot.as_slices();
        let ranked = rank_and_pick(
            &self.embedder,
            RankInput {
                topic: &topic,
                texts: &texts,
                tokens: &toks,
                sources: &sources,
                input_tokens: &tok.chunks,
                recent_bot,
                surprises: &surprises,
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
        self.log.append_turn(
            Record::user(input.to_string(), learned),
            Record::bot(
                chosen_text.clone(),
                learned,
                sim,
                ranked.slipped,
                path,
                novelty_lcs,
                chosen_tokens.len(),
            ),
        )?;
        if learned {
            self.absorb(Role::User, input, &tok.chunks);
            self.absorb(Role::Bot, &chosen_text, &chosen_tokens);
            self.pairs.push_live(
                &self.embedder,
                input.to_string(),
                tok.chunks.clone(),
                chosen_tokens.clone(),
                self.params.n_retrieve_scan,
            );
        }
        for &c in tok.chunks.iter().chain(chosen_tokens.iter()) {
            self.history.push_back(c);
        }
        trim_history(&mut self.history, self.params.l_max_capped() * 4);
        if self.params.self_window > 0 {
            self.recent_bot.push_back(chosen_text.clone());
            while self.recent_bot.len() > self.params.self_window {
                self.recent_bot.pop_front();
            }
        }
        self.last_trace = Some(trace.clone());
        self.last_path = Some(path);

        Ok(Reply {
            text: chosen_text,
            trace,
        })
    }

    /// Route, then let every gated source propose into one pool: trigger →
    /// retrieve → Markov → echo, with echo as the named last resort. RNG
    /// consumption order is part of the reproducibility contract — sources
    /// must keep proposing in this order.
    fn propose_all(
        &mut self,
        input: &str,
        tok: &Tokenized,
        topic: &[f32],
        input_emb: &[f32],
    ) -> (
        Pool,
        Vec<crate::explain::GenStep>,
        Option<crate::explain::TriggerTrace>,
        crate::route::RoutePlan,
    ) {
        let trigger_hit = self
            .triggers
            .match_one(&self.embedder, input_emb, &self.params);
        let retr_sim = self.bots.max_sim(topic, self.params.n_retrieve_scan);
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
                self.bots.propose(
                    &mut pool,
                    input,
                    topic,
                    route.n_retrieve,
                    self.params.mmr_lambda,
                    self.params.n_retrieve_scan,
                );
            }
            // Adapt (Reudy analog): deterministic, no RNG, so inserting it
            // here keeps the RNG-consumption contract of the sources below.
            if self.params.n_adapt > 0 && self.pairs.len() > 0 {
                self.pairs.propose(
                    &mut pool,
                    &self.intern,
                    &self.store,
                    input,
                    &tok.chunks,
                    input_emb,
                    topic,
                    self.params.n_adapt,
                    self.params.n_retrieve_scan,
                );
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
                self.propose_echo(&mut pool, tok);
            }
        }

        // Exact self-repeats of the newest three replies are dropped
        // outright: the soft penalty ranks them down, but when a whole pool
        // scores badly the least-bad repeat still wins. Only three, not the
        // full window — a hard ban over all of `self_window` starves a small
        // register (seed-scale band fell 67→42%); the soft penalty keeps
        // covering the rest. Triggers stay: repeating the dictionary is a
        // ritual, not a rut. The fallbacks below refill an emptied pool.
        if self.params.self_window > 0 {
            pool.items.retain(|p| {
                p.source == PathKind::Trigger
                    || !self.recent_bot.iter().rev().take(3).any(|r| r == &p.text)
            });
        }
        if pool.is_empty() {
            self.propose_echo(&mut pool, tok);
        }
        if pool.is_empty() {
            pool.push(PathKind::Echo, "…".into(), Vec::new());
        }
        (pool, steps, trigger_tr, route)
    }

    fn absorb(&mut self, role: Role, text: &str, chunks: &[TokenId]) {
        self.tokenizer.observe(text);
        if chunks.is_empty() {
            return;
        }
        self.store.push_utterance(chunks);
        if self.params.bidir {
            let rev: Vec<TokenId> = chunks.iter().rev().copied().collect();
            self.rev_store.push_utterance(&rev);
        }
        if role == Role::Bot {
            self.bots.push_live(
                &self.embedder,
                text.to_string(),
                chunks.to_vec(),
                self.params.n_retrieve_scan,
            );
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
        // MegaHAL analog: anchor on the rarest in-corpus content chunk of the
        // input and grow the reply in both directions around it. Half the
        // candidate slots try this when an anchor exists.
        let anchor = self.pick_anchor(user_chunks);
        let n_bi = if anchor.is_some() { n_cand / 2 } else { 0 };
        let mut memo = NextMemo::default();
        let mut memo_rev = NextMemo::default();
        let mut seen = FxHashSet::default();
        let mut attempts = 0;
        let start = pool.items.len();
        while pool.items.len() - start < n_cand && attempts < n_cand.max(1) * 4 {
            attempts += 1;
            let g = if pool.items.len() - start < n_bi {
                let anchor = anchor.expect("n_bi > 0 implies anchor");
                self.gen_anchored(anchor, kn, &mut memo, &mut memo_rev)
            } else {
                let uni = self.store.sampling_view(kn).expect("warmed above");
                generate_one(
                    &self.store,
                    self.smoothing.as_ref(),
                    &self.params,
                    &ctx_seed,
                    &parrot,
                    uni,
                    &mut memo,
                    &mut self.rng,
                )
            };
            let surprise = mean_surprise(&g.steps);
            let mut toks = g.tokens;
            trim_leading_punct(&self.intern, &mut toks);
            if toks.is_empty() {
                continue;
            }
            if !seen.insert(toks.clone()) {
                continue;
            }
            let chunk_strs: Vec<String> = toks
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
            pool.push_scored(PathKind::Markov, text, toks, surprise);
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

    /// Rarest in-corpus content chunk of the input; ties go to the later
    /// position (fresher topic). None when nothing usable is in the corpus.
    fn pick_anchor(&self, user_chunks: &[TokenId]) -> Option<TokenId> {
        if !self.params.bidir || self.store.is_empty() || self.rev_store.is_empty() {
            return None;
        }
        let mut best: Option<(u32, usize, TokenId)> = None;
        for (i, &id) in user_chunks.iter().enumerate() {
            if is_special(id) || crate::tokenizer::is_punct_str(self.intern.get(id)) {
                continue;
            }
            let c = self.store.count_of(id);
            if c == 0 {
                continue;
            }
            let better = match best {
                None => true,
                Some((bc, bi, _)) => c < bc || (c == bc && i > bi),
            };
            if better {
                best = Some((c, i, id));
            }
        }
        best.map(|(_, _, id)| id)
    }

    /// One keyword-anchored candidate: forward continuation after the anchor
    /// on the normal store, then leftward growth on the reversed-stream twin
    /// (a suffix walk there predicts the *preceding* chunk). MegaHAL's
    /// two-model trick with both models being the same SA machinery.
    fn gen_anchored(
        &mut self,
        anchor: TokenId,
        kn: bool,
        memo: &mut NextMemo,
        memo_rev: &mut NextMemo,
    ) -> Generated {
        self.store.warm_sampling(kn);
        self.rev_store.warm_sampling(kn);
        let uni = self.store.sampling_view(kn).expect("warmed above");
        let fwd = generate_one(
            &self.store,
            self.smoothing.as_ref(),
            &self.params,
            &[anchor],
            &[],
            uni,
            memo,
            &mut self.rng,
        );
        let mut seq = vec![anchor];
        seq.extend(fwd.tokens);
        let mut bparams = self.params.clone();
        bparams.max_gen_len = self.params.max_gen_len.saturating_sub(seq.len()).max(1);
        let rev_ctx: Vec<TokenId> = seq.iter().rev().copied().collect();
        let uni_rev = self.rev_store.sampling_view(kn).expect("warmed above");
        let bwd = generate_one(
            &self.rev_store,
            self.smoothing.as_ref(),
            &bparams,
            &rev_ctx,
            &[],
            uni_rev,
            memo_rev,
            &mut self.rng,
        );
        let mut tokens: Vec<TokenId> = bwd.tokens.iter().rev().copied().collect();
        tokens.extend(seq);
        let mut steps = fwd.steps;
        steps.extend(bwd.steps);
        Generated { tokens, steps }
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
        if self.params.bidir {
            self.rev_store.merge();
        }
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
        self.log.append(Record::meta(
            if good { "good".into() } else { "bad".into() },
            Some(path),
        ))?;
        Ok(format!(
            "pref {} {:?}  prior={:+.2}",
            if good { "good" } else { "bad" },
            path,
            self.path_prior[route::prior_index(path)]
        ))
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
        // Last four history tokens in stream order (an earlier version
        // reversed them by accident).
        let parrot: Vec<TokenId> = ctx[ctx.len().saturating_sub(4)..].to_vec();
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
        let owned: Vec<(Role, String, bool)> = self
            .log
            .records
            .iter()
            .map(|r| (r.role, r.text.clone(), r.learned))
            .collect();
        let recs: Vec<(Role, &str, bool)> = owned
            .iter()
            .map(|(role, t, learned)| (*role, t.as_str(), *learned))
            .collect();
        let mut tokenizer = Tokenizer::new(&self.params);
        let replayed = replay_speech(&self.params, &self.embedder, &mut tokenizer, &recs);
        self.tokenizer = tokenizer;
        self.intern = replayed.intern;
        self.store = replayed.store;
        self.topic = replayed.topic;
        self.history = replayed.history;
        self.bots = replayed.bots;
        self.recent_bot = replayed.recent_bot;
        self.pairs = replayed.pairs;
        self.rev_store = replayed.rev_store;
        smoothing::sync_to_store(self.smoothing.as_mut(), &self.params, &self.store);
        Ok(())
    }
}

fn apply_pref(prior: &mut [f32; 5], path: PathKind, good: bool, step: f32, clip: f32) {
    let i = route::prior_index(path);
    let d = if good { step } else { -step };
    let clip = clip.max(0.0);
    prior[i] = (prior[i] + d).clamp(-clip, clip);
}

/// Everything the replay of a JSONL log rebuilds. `open` and
/// `retokenize_from_log` share this so the two paths cannot drift.
struct Replayed {
    intern: Interner,
    store: Store,
    topic: TopicTracker,
    history: VecDeque<TokenId>,
    bots: BotStore,
    recent_bot: VecDeque<String>,
    pairs: PairStore,
    rev_store: Store,
}

/// MegaHAL's surprise: mean −ln p over the generation steps (None when no
/// step carried probability mass).
fn mean_surprise(steps: &[GenStep]) -> Option<f32> {
    let mut sum = 0.0f32;
    let mut n = 0u32;
    for s in steps {
        if s.p > 0.0 && s.logp.is_finite() {
            sum -= s.logp;
            n += 1;
        }
    }
    if n == 0 {
        None
    } else {
        Some(sum / n as f32)
    }
}

/// A generated reply must not open with punctuation (「、おはよう」): the
/// corpus stores 「、」 as its own chunk, so decode can start there. Leading
/// specials are dropped too (display already filtered them).
fn trim_leading_punct(intern: &Interner, toks: &mut Vec<TokenId>) {
    while let Some(&id) = toks.first() {
        if is_special(id) || crate::tokenizer::is_punct_str(intern.get(id)) {
            toks.remove(0);
        } else {
            break;
        }
    }
}

/// Train the tokenizer on learned records, then replay every speech record
/// into interner / store / history / bots. Store pushes are deferred and
/// merged once at the end — no lookups happen mid-replay, so the final state
/// is identical to per-utterance pushes without the quadratic SA rebuilds
/// (`Engine::open` used to rebuild every `merge_threshold` tokens).
///
/// The topic window takes only user records: that is what the live path
/// pushes (`respond` embeds the input, never the reply), and replay has to
/// agree or the window diverges after reopen. Only the last `k_topic` user
/// records can still be inside it, so earlier ones are not embedded.
fn replay_speech(
    params: &Params,
    embedder: &HashEmbedder,
    tokenizer: &mut Tokenizer,
    recs: &[(Role, &str, bool)],
) -> Replayed {
    for (role, text, learned) in recs {
        if *role == Role::Meta || !*learned {
            continue;
        }
        tokenizer.observe(text);
    }

    let mut intern = Interner::new();
    let mut store = Store::new(params.merge_threshold);
    let mut rev_store = Store::new(params.merge_threshold);
    let mut topic = TopicTracker::new(params.embed_dim, params.k_topic);
    let mut history = VecDeque::new();
    let mut bots = BotStore::default();
    let mut pairs = PairStore::default();
    let mut recent_bot: VecDeque<String> = VecDeque::new();
    // (user utterance, chunks) waiting for the reply that completes a pair.
    let mut pending_user: Option<(String, Vec<TokenId>)> = None;

    let n_user = recs
        .iter()
        .filter(|(role, _, _)| *role == Role::User)
        .count();
    let k_topic = params.k_topic.max(1);
    let mut user_idx = 0usize;
    for (role, text, learned) in recs {
        if *role == Role::Meta {
            continue;
        }
        let tok = tokenizer.tokenize(&mut intern, text);
        if *learned {
            store.push_utterance_deferred(&tok.chunks);
            if params.bidir {
                let rev: Vec<TokenId> = tok.chunks.iter().rev().copied().collect();
                rev_store.push_utterance_deferred(&rev);
            }
            if *role == Role::Bot {
                bots.push_raw(text.to_string(), tok.chunks.clone());
            }
        }
        match *role {
            Role::User => pending_user = Some((text.to_string(), tok.chunks.clone())),
            Role::Bot => {
                if let Some((ut, uc)) = pending_user.take() {
                    if *learned {
                        pairs.push_raw(ut, uc, tok.chunks.clone());
                    }
                }
            }
            Role::Meta => {}
        }
        if *role == Role::Bot && params.self_window > 0 {
            recent_bot.push_back(text.to_string());
            while recent_bot.len() > params.self_window {
                recent_bot.pop_front();
            }
        }
        for &c in &tok.chunks {
            history.push_back(c);
        }
        if *role == Role::User {
            if user_idx + k_topic >= n_user {
                let mut v = vec![0.0f32; params.embed_dim];
                embedder.embed(text, &mut v);
                topic.push(&v);
            }
            user_idx += 1;
        }
    }
    store.merge();
    if params.bidir {
        rev_store.merge();
    }
    trim_history(&mut history, params.l_max_capped() * 4);
    bots.finish(embedder, params.n_retrieve_scan);
    pairs.finish(embedder, params.n_retrieve_scan);

    Replayed {
        intern,
        store,
        topic,
        history,
        bots,
        recent_bot,
        pairs,
        rev_store,
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

    /// The adapt source proposes for inputs similar to a learned exchange,
    /// and the reversed twin store stays in lockstep with the forward one.
    #[test]
    fn adapt_proposes_and_rev_store_tracks() {
        let params = Params {
            p_learn: 1.0,
            p_slip: 0.0,
            ..Params::default()
        };
        let mut e = Engine::ephemeral(params, 21).unwrap();
        e.respond("コーヒー飲む？").unwrap();
        e.respond("散歩しよう").unwrap();
        assert_eq!(
            e.rev_store.len(),
            e.store.len(),
            "reversed twin must mirror the forward store"
        );
        let r = e.respond("紅茶飲む？").unwrap();
        assert!(
            r.trace
                .candidates
                .iter()
                .any(|c| c.source == PathKind::Adapt),
            "pool should contain an Adapt candidate: {:?}",
            r.trace
                .candidates
                .iter()
                .map(|c| (c.source, c.text.clone()))
                .collect::<Vec<_>>()
        );
    }

    /// Keyword-anchored bidirectional generation can place the anchor after
    /// generated left context — a left-to-right decoder from the user's rare
    /// chunk could never do that.
    #[test]
    fn anchored_generation_gives_anchor_left_context() {
        let params = Params {
            p_learn: 1.0,
            p_slip: 0.0,
            n_cand: 10,
            ..Params::default()
        };
        let mut e = Engine::ephemeral(params, 17).unwrap();
        for _ in 0..6 {
            e.respond("あか、あお、らくだ、みどり").unwrap();
        }
        let mut saw_left_context = false;
        for _ in 0..6 {
            let r = e.respond("らくだ").unwrap();
            for c in &r.trace.candidates {
                if c.source == PathKind::Markov
                    && c.text.contains("らくだ")
                    && !c.text.starts_with("らくだ")
                {
                    saw_left_context = true;
                }
            }
        }
        assert!(
            saw_left_context,
            "some markov candidate should contain the anchor mid-text"
        );
    }

    /// Non-trigger replies never exactly repeat one of the last
    /// `self_window` own replies (deterministic for a fixed seed).
    #[test]
    fn no_exact_self_repeat_within_window() {
        let params = Params {
            p_learn: 1.0,
            p_slip: 0.0,
            ..Params::default()
        };
        let mut e = Engine::ephemeral(params, 13).unwrap();
        let inputs = ["こんにちは", "散歩しよう", "猫かわいい"];
        let mut said: Vec<(String, PathKind)> = Vec::new();
        for i in 0..12 {
            let r = e.respond(inputs[i % inputs.len()]).unwrap();
            // The hard guarantee covers the newest three replies; farther
            // back only the soft penalty applies.
            let recent: Vec<&String> = said.iter().rev().take(3).map(|(t, _)| t).collect();
            if r.trace.path != PathKind::Trigger && r.trace.path != PathKind::Echo {
                assert!(
                    !recent.contains(&&r.text),
                    "turn {i}: repeated own reply {:?}",
                    r.text
                );
            }
            said.push((r.text, r.trace.path));
        }
    }

    /// Markov proposals never open with punctuation — the corpus keeps 「、」
    /// as its own chunk, so decode can otherwise start a reply there.
    #[test]
    fn markov_candidates_never_start_with_punctuation() {
        let params = Params {
            p_learn: 1.0,
            p_slip: 0.0,
            n_cand: 8,
            ..Params::default()
        };
        let mut e = Engine::ephemeral(params, 6).unwrap();
        let lines = [
            "うん、散歩しよう",
            "ね、コーヒー飲む？",
            "え、猫かわいいね",
            "あ、仕事わすれてた",
            "うん、そうだね",
            "ね、ゲームしよう",
        ];
        for line in lines.iter().chain(lines.iter()) {
            let r = e.respond(line).unwrap();
            for c in &r.trace.candidates {
                if c.source == PathKind::Markov {
                    let first = c.text.chars().next().unwrap_or('x');
                    assert!(
                        !matches!(first, '、' | '。' | '！' | '？' | '…' | '・'),
                        "markov candidate starts with punctuation: {}",
                        c.text
                    );
                }
            }
        }
    }

    /// The live path pushes only user inputs into the topic window; a reopen
    /// must restore the same window, not a user+bot mixture.
    #[test]
    fn topic_window_matches_after_reopen() {
        let dir = std::env::temp_dir().join(format!(
            "munou-topic-{}-{}",
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
        let live_topic;
        {
            let mut e = Engine::open(OpenConfig {
                params: params.clone(),
                seed: 7,
                log_path: Some(log.clone()),
                triggers_path: None,
            })
            .unwrap();
            e.respond("こんにちは").unwrap();
            e.respond("散歩しよう").unwrap();
            live_topic = e.stats().topic_window;
        }
        let e2 = Engine::open(OpenConfig {
            params,
            seed: 7,
            log_path: Some(log.clone()),
            triggers_path: None,
        })
        .unwrap();
        assert_eq!(
            e2.stats().topic_window,
            live_topic,
            "replayed topic window must match the live one (user inputs only)"
        );
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
