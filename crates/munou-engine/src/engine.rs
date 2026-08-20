use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rand_chacha::ChaCha8Rng;
use rand_core::{Rng, SeedableRng};
use rustc_hash::FxHashSet;

use crate::adapt::PairStore;
use crate::embed::{cosine, Embedder, HashEmbedder, TopicTracker};
use crate::error::Result;
use crate::eval::EvalAccum;
use crate::explain::{GenStep, PathKind, Trace};
use crate::generate::{generate_one, GenCaches, Generated};
use crate::ids::{is_special, TokenId};
use crate::interest::InterestLedger;
use crate::interject::InterjectBank;
use crate::intern::Interner;
use crate::log::{AppendLog, Record, Role};
use crate::milestone;
use crate::mix::Pool;
use crate::observe::{LogDigest, Observe};
use crate::params::{MixMode, Params, SmoothingKind};
use crate::retrieve::BotStore;
use crate::route;
use crate::select::{rank_and_pick, RankInput};
use crate::smoothing::{self, Smoothing};
use crate::store::Store;
use crate::tokenizer::{detokenize, Tokenized, Tokenizer};
use crate::trigger::TriggerDict;
use crate::weather;

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
    /// 合いの手 — a short learned line printed before `text`. Display only:
    /// not logged, not absorbed (replay stays exact).
    pub interject: Option<String>,
    /// 節目 — a growth mark crossed by this turn (counts and days, not fake
    /// emotion). Derived from the log digest, so it never re-fires on replay.
    pub milestone: Option<String>,
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
    /// Adapt source: learned (user → reply) exchanges.
    pairs: PairStore,
    /// Reversed-stream twin of `store` for keyword-anchored bidirectional
    /// generation. Empty when `bidir` is off.
    rev_store: Store,
    /// Closed analog of RLHF: additive path prior from `/good` `/bad`.
    path_prior: [f32; 5],
    /// Last bot path, restored from the log so `/good` works after reopen.
    last_path: Option<PathKind>,
    /// Incremental log digest: counts/paths/last-bot for stats and gauges,
    /// updated per append instead of rescanning the whole log per turn.
    digest: LogDigest,
    /// Cross-turn generation caches (counts + frozen distributions), valid
    /// until the corpus changes: absorb clears, non-learning turns reuse.
    gen_caches: GenCaches,
    gen_caches_rev: GenCaches,
    /// Last `self_window` own reply texts for the self-repetition penalty.
    /// Text, not tokens: the penalty is char-level so live and replay agree
    /// exactly (token ids drift across reopen).
    recent_bot: VecDeque<String>,
    /// 関心 — dual-timescale chunk weights on the log-position clock.
    interest: InterestLedger,
    /// 合いの手 bank — short learned lines with frequencies.
    interjects: InterjectBank,
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
            interest,
            interjects,
        } = replayed;

        let mut eval = EvalAccum::default();
        for rec in &log.records {
            eval.ingest_bot(rec, &cfg.params);
        }
        let digest = LogDigest::scan(&log.records);

        let mut smoothing = smoothing::boxed(cfg.params.smoothing, cfg.params.kn_discount);
        smoothing::sync_to_store(smoothing.as_mut(), &cfg.params, &store);
        let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
        if let Some(pos) = log
            .records
            .iter()
            .rev()
            .find(|rec| rec.role == Role::Bot)
            .and_then(Record::saved_rng_word_pos)
        {
            rng.set_word_pos(pos);
        }
        Ok(Self {
            intern,
            tokenizer,
            store,
            embedder,
            topic,
            triggers,
            rng,
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
            digest,
            gen_caches: GenCaches::default(),
            gen_caches_rev: GenCaches::default(),
            recent_bot,
            interest,
            interjects,
        })
    }

    pub fn load_triggers(&mut self, path: &Path) -> Result<()> {
        self.triggers = TriggerDict::from_path(path)?;
        self.triggers.warm(&self.embedder);
        Ok(())
    }

    pub fn respond(&mut self, input: &str) -> Result<Reply> {
        let t0 = Instant::now();
        self.gen_caches.trim();
        self.gen_caches_rev.trim();
        let input = input.trim();
        let tok = self.tokenizer.tokenize(&mut self.intern, input);

        // 日和: the day is the UTC day of the *previous* log record — never
        // the wall clock at scoring time — so a reply stays a pure function
        // of (log, seed, input) and replays reproduce the same weather.
        let day = self.digest.last_speech_t.map(weather::day_of_ms);
        let wthr = if self.params.weather {
            day.map(|d| weather::day_weather(self.seed, d))
                .unwrap_or(&weather::CALM)
        } else {
            &weather::CALM
        };
        let aloof = self.digest.aloof_left > 0;
        let gains = weather::effective(wthr, aloof);
        // 口をつく: one draw per turn, before the sources (the RNG
        // consumption order is part of the reproducibility contract).
        let release_roll: f64 = crate::rng::rand_f64(&mut self.rng);
        let release = release_roll < (self.params.hearsay_release * gains.release).clamp(0.0, 1.0);
        let care = day.and_then(|d| self.care_word(d));

        let mut q = vec![0.0f32; self.embedder.dim()];
        self.embedder.embed(input, &mut q);
        self.topic.push(&q);
        let mut topic = vec![0.0f32; self.embedder.dim()];
        self.topic.mean(&mut topic);

        let (pool, steps, trigger_tr, route) = self.propose_all(input, &tok, &topic, &q, release);

        let texts = pool.texts();
        let toks = pool.tokens();
        let sources = pool.sources();
        let surprises = pool.surprises();

        // 関心 + 気になる語: additive selection term per candidate. Hearsay
        // chunks carry no interest (score() is None for them).
        let bonus: Vec<f32> = toks
            .iter()
            .zip(texts.iter())
            .map(|(tk, tx)| {
                let mut b = 0.0f32;
                if self.params.interest_weight != 0.0 {
                    let mut best = 0.0f32;
                    for &id in tk.iter() {
                        if is_special(id) {
                            continue;
                        }
                        if let Some(s) = self.interest.score(id, self.params.hearsay_min) {
                            best = best.max(s);
                        }
                    }
                    b += self.params.interest_weight * best;
                }
                if let Some(cw) = &care {
                    if gains.care != 0.0 && tx.contains(cw.as_str()) {
                        b += self.params.care_bonus * gains.care;
                    }
                }
                b
            })
            .collect();

        // 日和は slip の量だけを動かす（帯域・減点はそのまま）。
        let mut eff = self.params.clone();
        eff.p_slip = (eff.p_slip * gains.slip).clamp(0.0, 1.0);

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
                bonus: &bonus,
            },
            &eff,
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

        let learn_roll: f64 = crate::rng::rand_f64(&mut self.rng);
        let learned = learn_roll < self.params.p_learn.clamp(0.0, 1.0);

        // 合いの手: RNG consumption is fixed — one roll when the bank is
        // live, one pick when the roll fires. Replies of ≤4 chars skip it
        // (an interjection before an interjection-length reply doubles up).
        let interject = if self.params.interject_rate > 0.0
            && self.interjects.distinct() >= crate::interject::MIN_DISTINCT
        {
            let roll: f64 = crate::rng::rand_f64(&mut self.rng);
            if roll < (self.params.interject_rate * gains.interject).clamp(0.0, 1.0)
                && chosen_text.chars().count() > 4
            {
                self.interjects.pick(&mut self.rng, &chosen_text)
            } else {
                None
            }
        } else {
            None
        };

        let weather_note = if self.params.weather && day.is_some() {
            let mut s = format!(
                "{} slip×{:.2} 合いの手×{:.2} 口をつく×{:.2}",
                wthr.name, gains.slip, gains.interject, gains.release
            );
            if let Some(cw) = &care {
                s.push_str(&format!(" 気になる語「{cw}」"));
            }
            if aloof {
                s.push_str("（よそよそしい）");
            }
            Some(s)
        } else {
            None
        };

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
            p_slip: eff.p_slip,
            learned: learned && !(tok.chunks.is_empty() && chosen_tokens.is_empty()),
            learn_roll,
            p_learn: self.params.p_learn,
            steps,
            elapsed_us: t0.elapsed().as_micros(),
            novelty_lcs,
            similarity: sim,
            band_hit,
            route: Some(route.explain_line()),
            path_prior: self.path_prior,
            weather: weather_note,
            interject: interject.clone(),
        };

        // Per-side learned flags: an empty chunk list absorbs nothing live,
        // so the record must not claim absorption either — otherwise replay
        // re-tokenises the text (e.g. the ellipsis fallback) into a corpus
        // the live process never had.
        let learned_user = learned && !tok.chunks.is_empty();
        let learned_bot = learned && !chosen_tokens.is_empty();
        let rec_user = Record::user(input.to_string(), learned_user);
        let mut rec_bot = Record::bot(
            chosen_text.clone(),
            learned_bot,
            sim,
            ranked.slipped,
            path,
            novelty_lcs,
            chosen_tokens.len(),
        );
        rec_bot.set_rng_word_pos(self.rng.get_word_pos());
        // Append first: the log is the source of truth, and digest/eval must
        // not advance when the write fails. 節目 is a digest crossing, so the
        // pre-append values are captured here.
        let pre_learned = self.digest.learned;
        let pre_last_t = self.digest.last_speech_t;
        let pre_aloof = self.digest.aloof_left;
        self.log.append_turn(rec_user, rec_bot)?;
        let n = self.log.records.len();
        self.digest.ingest(&self.log.records[n - 2].clone());
        self.digest.ingest(&self.log.records[n - 1].clone());
        let milestone = milestone::lines(pre_learned, pre_last_t, pre_aloof, &self.digest)
            .into_iter()
            .next();
        self.eval.observe(&trace, chosen_tokens.len());
        if learned_user {
            self.absorb(Role::User, input, &tok.chunks);
        }
        if learned_bot {
            self.absorb(Role::Bot, &chosen_text, &chosen_tokens);
            if self.params.n_adapt > 0 {
                self.pairs.push_live(
                    &self.embedder,
                    input.to_string(),
                    tok.chunks.clone(),
                    chosen_tokens.clone(),
                    self.params.n_retrieve_scan,
                );
            }
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
            interject,
            milestone,
            trace,
        })
    }

    /// きょうの気になる語: a deterministic pick among established (non-hearsay)
    /// chunks, keyed by (seed, day). Sorted by surface string so the pick
    /// survives reopen (token ids drift, text does not).
    fn care_word(&self, day: u64) -> Option<String> {
        if !self.params.weather || self.params.care_bonus == 0.0 {
            return None;
        }
        let ids = self.interest.established(self.params.hearsay_min);
        let mut words: Vec<&str> = ids
            .iter()
            .map(|id| self.intern.get(*id))
            .filter(|s| !crate::tokenizer::is_punct_str(s) && s.chars().count() >= 2)
            .collect();
        if words.is_empty() {
            return None;
        }
        words.sort_unstable();
        words.dedup();
        Some(words[weather::care_index(self.seed, day, words.len())].to_string())
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
        release: bool,
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
            // Adapt: deterministic, no RNG, so inserting it
            // here keeps the RNG-consumption contract of the sources below.
            // Pool mode only — exclusive is the v0.1 XOR contract.
            if self.params.mix == MixMode::Pool && self.params.n_adapt > 0 && self.pairs.len() > 0 {
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
                self.propose_markov(&mut pool, &tok.chunks, &mut steps, n_cand, release);
            }
            if route.run_echo {
                self.propose_echo(&mut pool, tok);
            }
        }

        // Exact self-repeats of the newest three replies are dropped
        // outright: the soft penalty ranks them down, but when a whole pool
        // scores badly the least-bad repeat still wins. Only three, not the
        // full window — a hard ban over all of `self_window` drops the
        // seed-scale band 67→42%; the soft penalty covers the rest. Triggers
        // are exempt: dictionary responses are expected to repeat. The
        // fallbacks below refill an emptied pool.
        if self.params.self_window > 0 && self.params.self_penalty > 0.0 {
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
        // 体験層: 関心の帳簿と合いの手バンクは、吸収された発話だけを食べる
        // （replay と同じ条件・同じ順序）。
        {
            let intern = &self.intern;
            self.interest
                .learn(chunks, |id| crate::tokenizer::is_punct_str(intern.get(id)));
        }
        self.interjects.learn(text);
        self.gen_caches.clear();
        self.gen_caches_rev.clear();
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
        release: bool,
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
        // Anchor on the rarest in-corpus content chunk of the
        // input and grow the reply in both directions around it. Half the
        // candidate slots try this when an anchor exists.
        let anchor = self.pick_anchor(user_chunks, release);
        let n_bi = if anchor.is_some() { n_cand / 2 } else { 0 };
        let mut seen = FxHashSet::default();
        let mut attempts = 0;
        let start = pool.items.len();
        while pool.items.len() - start < n_cand && attempts < n_cand.max(1) * 4 {
            attempts += 1;
            let g = if pool.items.len() - start < n_bi {
                let anchor = anchor.expect("n_bi > 0 implies anchor");
                self.gen_anchored(anchor, kn)
            } else {
                let uni = self.store.sampling_view(kn).expect("warmed above");
                generate_one(
                    &self.store,
                    self.smoothing.as_ref(),
                    &self.params,
                    &ctx_seed,
                    &parrot,
                    uni,
                    &mut self.gen_caches,
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
    /// 聞きかじり (heard in fewer than `hearsay_min` utterances) does not
    /// anchor — a one-off typo must not anchor a reply — except on
    /// 口をつく turns (`release`), when it may slip out.
    fn pick_anchor(&self, user_chunks: &[TokenId], release: bool) -> Option<TokenId> {
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
            if !release && self.interest.is_hearsay(id, self.params.hearsay_min) {
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
    /// (a suffix walk there predicts the *preceding* chunk).
    fn gen_anchored(&mut self, anchor: TokenId, kn: bool) -> Generated {
        self.store.warm_sampling(kn);
        self.rev_store.warm_sampling(kn);
        let uni = self.store.sampling_view(kn).expect("warmed above");
        // The anchor takes one slot of the hard cap; forward gets the rest
        // and backward whatever remains — the total never exceeds
        // max_gen_len.
        let mut fparams = self.params.clone();
        fparams.max_gen_len = self.params.max_gen_len.saturating_sub(1);
        let fwd = generate_one(
            &self.store,
            self.smoothing.as_ref(),
            &fparams,
            &[anchor],
            &[],
            uni,
            &mut self.gen_caches,
            &mut self.rng,
        );
        let mut seq = vec![anchor];
        seq.extend(fwd.tokens);
        let budget = self.params.max_gen_len.saturating_sub(seq.len());
        let mut steps = fwd.steps;
        let mut tokens: Vec<TokenId> = Vec::new();
        if budget > 0 {
            let mut bparams = self.params.clone();
            bparams.max_gen_len = budget;
            let rev_ctx: Vec<TokenId> = seq.iter().rev().copied().collect();
            let uni_rev = self.rev_store.sampling_view(kn).expect("warmed above");
            let bwd = generate_one(
                &self.rev_store,
                self.smoothing.as_ref(),
                &bparams,
                &rev_ctx,
                &[],
                uni_rev,
                &mut self.gen_caches_rev,
                &mut self.rng,
            );
            tokens = bwd.tokens.iter().rev().copied().collect();
            steps.extend(bwd.steps);
        }
        tokens.extend(seq);
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
        let rec = Record::meta(if good { "good".into() } else { "bad".into() }, Some(path));
        self.log.append(rec)?;
        let last = self.log.records.last().expect("just appended").clone();
        self.digest.ingest(&last);
        Ok(format!(
            "pref {} {:?}  prior={:+.2}",
            if good { "good" } else { "bad" },
            path,
            self.path_prior[route::prior_index(path)]
        ))
    }

    pub fn stats(&self) -> Stats {
        let episodic = self.digest.speech;
        let meta = self.digest.meta;
        let learned = self.digest.learned;
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

    /// あゆみ — the one-page record of this individual: birth day, 初語,
    /// counts, marks, today's 日和, and the current 関心 top words. Every
    /// number is derived from the log; nothing here is a mood model.
    pub fn ayumi_text(&self) -> String {
        let s = self.stats();
        let d = &self.digest;
        let mut out = String::new();
        out.push_str("あゆみ — 人工無脳君\n");
        match (d.first_speech_t, d.last_speech_t) {
            (Some(f), Some(l)) => {
                let fd = weather::day_of_ms(f);
                let (y, m, dd) = weather::civil_from_day(fd);
                let age = weather::day_of_ms(l).saturating_sub(fd);
                out.push_str(&format!(
                    "うまれた日  {y:04}-{m:02}-{dd:02}（{}日目）\n",
                    age + 1
                ));
                if let Some(fw) = &d.first_learned_user {
                    out.push_str(&format!("初語        「{fw}」\n"));
                }
                let (lm, dm) = milestone::achieved(d.learned, age);
                if !lm.is_empty() || !dm.is_empty() {
                    let lms: Vec<String> = lm.iter().map(|m| format!("吸収{m}")).collect();
                    let dms: Vec<String> = dm.iter().map(|m| format!("{m}日")).collect();
                    out.push_str(&format!(
                        "節目        {}\n",
                        lms.into_iter().chain(dms).collect::<Vec<_>>().join(" ")
                    ));
                }
                if self.params.weather {
                    let day = weather::day_of_ms(l);
                    let w = weather::day_weather(self.seed, day);
                    let care = self
                        .care_word(day)
                        .map(|c| format!("  気になる語「{c}」"))
                        .unwrap_or_default();
                    let aloof = if d.aloof_left > 0 {
                        "（よそよそしい）"
                    } else {
                        ""
                    };
                    out.push_str(&format!("日和        {}{care}{aloof}\n", w.name));
                }
            }
            _ => out.push_str("うまれた日  （まだ記録なし）\n"),
        }
        out.push_str(&format!(
            "発話 {}  吸収 {}  tokens {}  vocab {}\n",
            s.utterances, s.learned, s.tokens, s.vocab
        ));
        let p = &d.paths;
        if d.path_known > 0 {
            out.push_str(&format!(
                "経路        trig={} mark={} retr={} echo={} adpt={}\n",
                p[0], p[1], p[2], p[3], p[4]
            ));
        }
        let mut tops: Vec<(String, f32)> = self
            .interest
            .established(self.params.hearsay_min)
            .into_iter()
            .filter_map(|id| {
                self.interest
                    .score(id, self.params.hearsay_min)
                    .map(|sc| (self.intern.get(id).to_string(), sc))
            })
            .filter(|(w, _)| !crate::tokenizer::is_punct_str(w))
            .collect();
        tops.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        tops.truncate(5);
        if !tops.is_empty() {
            let line: Vec<String> = tops.iter().map(|(w, sc)| format!("{w}({sc:.2})")).collect();
            out.push_str(&format!("関心        {}\n", line.join(" ")));
        }
        out
    }

    pub fn observe(&self) -> Observe {
        Observe::from_parts(
            &self.stats(),
            &self.params,
            &self.digest,
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
        self.gen_caches.trim();
        let ctx: Vec<TokenId> = self.history.iter().copied().collect();
        // Last four history tokens, in stream order.
        let parrot: Vec<TokenId> = ctx[ctx.len().saturating_sub(4)..].to_vec();
        let kn = matches!(self.params.smoothing, SmoothingKind::Kn);
        self.store.warm_sampling(kn);
        let uni = self.store.sampling_view(kn).expect("warmed above");
        let mut rng = self.rng.clone();
        generate_one(
            &self.store,
            self.smoothing.as_ref(),
            &self.params,
            &ctx,
            &parrot,
            uni,
            &mut self.gen_caches,
            &mut rng,
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
        self.interest = replayed.interest;
        self.interjects = replayed.interjects;
        self.gen_caches.clear();
        self.gen_caches_rev.clear();
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
    interest: InterestLedger,
    interjects: InterjectBank,
}

/// Surprise: mean −ln p over the generation steps (None when no
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
/// is identical to per-utterance pushes without quadratic SA rebuilds.
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
    // The closed log repeats lines heavily; observe each distinct text once
    // with its multiplicity (bit-identical counts, one hash pass per line).
    {
        let mut weights: rustc_hash::FxHashMap<&str, u32> = rustc_hash::FxHashMap::default();
        let mut order: Vec<&str> = Vec::new();
        for (role, text, learned) in recs {
            if *role == Role::Meta || !*learned {
                continue;
            }
            let w = weights.entry(text).or_insert(0);
            if *w == 0 {
                order.push(text);
            }
            *w += 1;
        }
        for t in order {
            tokenizer.observe_n(t, weights[t]);
        }
    }

    let mut intern = Interner::new();
    let mut store = Store::new(params.merge_threshold);
    let mut rev_store = Store::new(params.merge_threshold);
    let mut topic = TopicTracker::new(params.embed_dim, params.k_topic);
    let mut history = VecDeque::new();
    let mut bots = BotStore::default();
    let mut pairs = PairStore::default();
    let mut recent_bot: VecDeque<String> = VecDeque::new();
    let mut interest = InterestLedger::default();
    let mut interjects = InterjectBank::default();
    // (user utterance, chunks) waiting for the reply that completes a pair.
    let mut pending_user: Option<(String, Vec<TokenId>)> = None;

    let n_user = recs
        .iter()
        .filter(|(role, _, _)| *role == Role::User)
        .count();
    let k_topic = params.k_topic.max(1);
    let mut user_idx = 0usize;
    // Tokenisation is a pure function during replay (the entropy model is
    // fully trained above and frozen), so repeated texts tokenise once.
    let mut tok_memo: rustc_hash::FxHashMap<&str, Tokenized> = rustc_hash::FxHashMap::default();
    for (role, text, learned) in recs {
        if *role == Role::Meta {
            continue;
        }
        let tok = tok_memo
            .entry(text)
            .or_insert_with(|| tokenizer.tokenize(&mut intern, text))
            .clone();
        if *learned {
            store.push_utterance_deferred(&tok.chunks);
            if params.bidir {
                let rev: Vec<TokenId> = tok.chunks.iter().rev().copied().collect();
                rev_store.push_utterance_deferred(&rev);
            }
            if *role == Role::Bot {
                bots.push_raw(text.to_string(), tok.chunks.clone());
            }
            if !tok.chunks.is_empty() {
                interest.learn(&tok.chunks, |id| {
                    crate::tokenizer::is_punct_str(intern.get(id))
                });
                interjects.learn(text);
            }
        }
        match *role {
            Role::User => pending_user = Some((text.to_string(), tok.chunks.clone())),
            Role::Bot => {
                if let Some((ut, uc)) = pending_user.take() {
                    if *learned && params.n_adapt > 0 {
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
        interest,
        interjects,
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

fn parrot_variant<R: Rng + ?Sized>(chunks: &[String], rng: &mut R) -> String {
    if chunks.is_empty() {
        return "…".into();
    }
    let mut v = chunks.to_vec();
    // Fisher–Yates, but keep it mild: swap a couple of neighbours
    if v.len() > 1 {
        let i = crate::rng::rand_below(rng, v.len() - 1);
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
        // weather=false: the 日和 dials depend on the current UTC day, so a
        // fixed seed pair can coincide on some days. This test targets the
        // seed-split of the RNG stream and must be date-free.
        let params = Params {
            weather: false,
            p_slip: 1.0,
            ..Params::default()
        };
        let mut a = Engine::ephemeral(params.clone(), 1).unwrap();
        let mut b = Engine::ephemeral(params, 2).unwrap();
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
    fn rng_position_matches_after_reopen() {
        let dir = std::env::temp_dir().join(format!(
            "munou-rng-reopen-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let log = dir.join("log.jsonl");
        crate::fabricate::write_jsonl(
            &log,
            crate::fabricate::FabricateOpts {
                pairs: 64,
                rng_seed: 1,
                unique_frac: 0.0,
            },
        )
        .unwrap();
        let params = Params {
            p_learn: 0.0,
            p_slip: 0.0,
            weather: false,
            interject_rate: 0.0,
            ..Params::default()
        };
        let mut live = Engine::open(OpenConfig {
            params: params.clone(),
            seed: 1,
            log_path: Some(log.clone()),
            triggers_path: None,
        })
        .unwrap();
        live.respond("こんにちは").unwrap();
        let live_pos = live.rng.get_word_pos();
        let resumed_log = dir.join("resumed.jsonl");
        std::fs::copy(&log, &resumed_log).unwrap();
        let mut reopened = Engine::open(OpenConfig {
            params,
            seed: 1,
            log_path: Some(resumed_log),
            triggers_path: None,
        })
        .unwrap();
        assert_eq!(reopened.rng.get_word_pos(), live_pos);
        let live_reply = live.respond("散歩しない？").unwrap();
        let reopened_reply = reopened.respond("散歩しない？").unwrap();
        assert_eq!(reopened_reply.text, live_reply.text);
        assert_eq!(reopened_reply.interject, live_reply.interject);
        assert_eq!(reopened_reply.trace.path, live_reply.trace.path);
        assert_eq!(reopened_reply.trace.learn_roll, live_reply.trace.learn_roll);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn markov_draw_does_not_advance_conversation_rng() {
        let mut e = Engine::ephemeral(Params::default(), 1).unwrap();
        e.respond("こんにちは").unwrap();
        let before = e.rng.get_word_pos();
        let _ = e.markov_draw();
        assert_eq!(e.rng.get_word_pos(), before);
    }

    #[test]
    fn latest_legacy_bot_does_not_reuse_stale_rng_position() {
        let dir = std::env::temp_dir().join(format!(
            "munou-rng-legacy-{}-{}",
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
                seed: 1,
                log_path: Some(log.clone()),
                triggers_path: None,
            })
            .unwrap();
            e.respond("こんにちは").unwrap();
        }
        {
            let mut append = AppendLog::open(Some(&log)).unwrap();
            append
                .append_turn(
                    Record::user("legacy user".into(), false),
                    Record::bot("legacy bot".into(), false, 0.0, false, PathKind::Echo, 0, 1),
                )
                .unwrap();
        }
        let reopened = Engine::open(OpenConfig {
            params: Params::default(),
            seed: 1,
            log_path: Some(log),
            triggers_path: None,
        })
        .unwrap();
        assert_eq!(reopened.rng.get_word_pos(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Exclusive mode is the v0.1 XOR: no Adapt beside a trigger-less pool.
    #[test]
    fn exclusive_mode_has_no_adapt_candidates() {
        let params = Params {
            mix: MixMode::Exclusive,
            p_learn: 1.0,
            p_slip: 0.0,
            ..Params::default()
        };
        let mut e = Engine::ephemeral(params, 5).unwrap();
        e.respond("コーヒー飲む？").unwrap();
        e.respond("散歩しよう").unwrap();
        let r = e.respond("紅茶飲む？").unwrap();
        assert!(
            r.trace
                .candidates
                .iter()
                .all(|c| c.source != PathKind::Adapt),
            "exclusive pool must not contain Adapt: {:?}",
            r.trace
                .candidates
                .iter()
                .map(|c| c.source)
                .collect::<Vec<_>>()
        );
    }

    /// max_gen_len is a hard cap, including for anchored bidirectional
    /// generation.
    #[test]
    fn max_gen_len_is_a_hard_cap() {
        let params = Params {
            p_learn: 1.0,
            p_slip: 0.0,
            max_gen_len: 3,
            ..Params::default()
        };
        let mut e = Engine::ephemeral(params, 17).unwrap();
        for _ in 0..6 {
            e.respond("あか、あお、らくだ、みどり、きいろ").unwrap();
        }
        for _ in 0..6 {
            let r = e.respond("らくだ").unwrap();
            for c in &r.trace.candidates {
                if c.source == PathKind::Markov {
                    assert!(
                        c.tokens.len() <= 3,
                        "markov candidate exceeds max_gen_len: {:?}",
                        c.tokens
                    );
                }
            }
        }
    }

    /// n_adapt=0 must switch the pair memory off entirely: no growth, no
    /// per-turn embedding.
    #[test]
    fn n_adapt_zero_disables_pair_store() {
        let params = Params {
            n_adapt: 0,
            p_learn: 1.0,
            p_slip: 0.0,
            ..Params::default()
        };
        let mut e = Engine::ephemeral(params, 3).unwrap();
        for line in ["こんにちは", "散歩しよう", "猫かわいい"] {
            e.respond(line).unwrap();
        }
        assert_eq!(e.pairs.len(), 0, "pair store must stay empty at n_adapt=0");
    }

    /// The empty-input fallback「…」must not diverge between live and reopen:
    /// live absorbs nothing (empty chunks), so the record must say so.
    #[test]
    fn empty_input_state_matches_after_reopen() {
        let dir = std::env::temp_dir().join(format!(
            "munou-empty-{}-{}",
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
        let live;
        {
            let mut e = Engine::open(OpenConfig {
                params: params.clone(),
                seed: 9,
                log_path: Some(log.clone()),
                triggers_path: None,
            })
            .unwrap();
            e.respond("").unwrap();
            live = (e.stats().tokens, e.bots.len());
        }
        let e2 = Engine::open(OpenConfig {
            params,
            seed: 9,
            log_path: Some(log.clone()),
            triggers_path: None,
        })
        .unwrap();
        // vocab/hist may still drift (bot text is re-tokenised on replay —
        // the documented asymmetry); the corpus and the retrieve store must
        // not.
        assert_eq!(
            (e2.stats().tokens, e2.bots.len()),
            live,
            "reopen must not grow a corpus the live process never had"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A torn final line (crash window) must not swallow the next record.
    #[test]
    fn torn_log_line_does_not_swallow_next_record() {
        let dir = std::env::temp_dir().join(format!(
            "munou-torn-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let log = dir.join("log.jsonl");
        std::fs::write(&log, "{\"v\":1,\"t\":1,\"role\":\"user\",\"te").unwrap();
        let params = Params {
            p_learn: 1.0,
            p_slip: 0.0,
            ..Params::default()
        };
        {
            let mut e = Engine::open(OpenConfig {
                params: params.clone(),
                seed: 3,
                log_path: Some(log.clone()),
                triggers_path: None,
            })
            .unwrap();
            e.respond("猫かわいい").unwrap();
        }
        let e2 = Engine::open(OpenConfig {
            params,
            seed: 3,
            log_path: Some(log.clone()),
            triggers_path: None,
        })
        .unwrap();
        assert_eq!(
            e2.stats().utterances,
            2,
            "user+bot must both survive a torn predecessor line"
        );
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

    /// 合いの手 is deterministic for a fixed seed and fires once the bank is
    /// live (three distinct short learned lines) at rate 1.
    #[test]
    fn interject_beat_fires_and_matches_between_same_seeds() {
        // weather=false: on a しめり day the dial scales rate 1.0 down to 0.4
        // and the firing assertion becomes date-dependent.
        let params = Params {
            p_learn: 1.0,
            p_slip: 0.0,
            interject_rate: 1.0,
            weather: false,
            ..Params::default()
        };
        let mut a = Engine::ephemeral(params.clone(), 11).unwrap();
        let mut b = Engine::ephemeral(params, 11).unwrap();
        let lines = [
            "はい",
            "うん",
            "おお",
            "今日はとてもいい天気だね",
            "散歩にいこうよ",
            "コーヒーのむ？",
        ];
        let mut fired = false;
        for l in lines.iter().chain(lines.iter()) {
            let ra = a.respond(l).unwrap();
            let rb = b.respond(l).unwrap();
            assert_eq!(ra.text, rb.text, "l={l}");
            assert_eq!(ra.interject, rb.interject, "l={l}");
            if ra.interject.is_some() {
                fired = true;
            }
        }
        assert!(
            fired,
            "rate=1 with a live bank must produce an interjection"
        );
    }

    /// 節目 fires exactly on the crossing turn and never again.
    #[test]
    fn milestone_fires_on_learned_crossing() {
        let params = Params {
            p_learn: 1.0,
            p_slip: 0.0,
            ..Params::default()
        };
        let mut e = Engine::ephemeral(params, 5).unwrap();
        let mut got = Vec::new();
        for i in 0..6 {
            let r = e.respond(&format!("こんにちは{i}")).unwrap();
            if let Some(m) = r.milestone {
                got.push((i, m));
            }
        }
        // 2 learned records per turn → the 吸収10 crossing is turn index 4.
        assert_eq!(got, vec![(4usize, "節目 吸収10".to_string())]);
    }

    /// The 合いの手 bank is replay-derived: reopening must reconstruct the
    /// same (text, count) sequence the live process built.
    #[test]
    fn interject_bank_survives_reopen() {
        let dir = std::env::temp_dir().join(format!(
            "munou-beat-{}-{}",
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
        let live;
        {
            let mut e = Engine::open(OpenConfig {
                params: params.clone(),
                seed: 9,
                log_path: Some(log.clone()),
                triggers_path: None,
            })
            .unwrap();
            for l in ["はい", "うん", "おお", "はい"] {
                e.respond(l).unwrap();
            }
            live = e.interjects.entries();
        }
        let e2 = Engine::open(OpenConfig {
            params,
            seed: 9,
            log_path: Some(log.clone()),
            triggers_path: None,
        })
        .unwrap();
        assert_eq!(e2.interjects.entries(), live);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 聞きかじり must not anchor while hearsay (release=false); 口をつく
    /// (release=true) lifts the gate.
    #[test]
    fn hearsay_gate_controls_anchoring() {
        let params = Params {
            p_learn: 1.0,
            p_slip: 0.0,
            hearsay_release: 0.0,
            ..Params::default()
        };
        let mut e = Engine::ephemeral(params, 17).unwrap();
        e.respond("あか、あお、らくだ、みどり").unwrap();
        let tok = e.tokenizer.tokenize(&mut e.intern, "らくだ");
        let id = *tok.chunks.first().expect("らくだ tokenises");
        if e.store.count_of(id) > 0 {
            let hearsay = e.interest.is_hearsay(id, e.params.hearsay_min);
            let gated = e.pick_anchor(&tok.chunks, false);
            if hearsay {
                assert_eq!(gated, None, "hearsay chunk must not anchor");
            }
            assert!(
                e.pick_anchor(&tok.chunks, true).is_some(),
                "release must lift the gate for an in-corpus chunk"
            );
        }
    }

    #[test]
    fn ayumi_reports_birth_and_first_word() {
        let params = Params {
            p_learn: 1.0,
            p_slip: 0.0,
            ..Params::default()
        };
        let mut e = Engine::ephemeral(params, 7).unwrap();
        e.respond("はじめてのあいさつ").unwrap();
        let a = e.ayumi_text();
        assert!(a.contains("うまれた日"), "{a}");
        assert!(a.contains("初語"), "{a}");
        assert!(a.contains("はじめてのあいさつ"), "{a}");
        assert!(a.contains("発話 2"), "{a}");
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
