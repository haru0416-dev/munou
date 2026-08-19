use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::SeedableRng;
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
use crate::params::Params;
use crate::select::rank_and_pick;
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
            tokenizer.observe(&rec.text);
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
        let dim = cfg.params.embed_dim;

        for rec in &log.records {
            let tok = tokenizer.tokenize(&mut intern, &rec.text);
            store.push_utterance(&tok.chunks);
            prior.push(tok.chunks.clone());
            for &c in &tok.chunks {
                history.push_back(c);
            }
            let mut v = vec![0.0f32; dim];
            embedder.embed(&rec.text, &mut v);
            topic.push(&v);
        }
        store.merge();
        trim_history(&mut history, cfg.params.l_max_capped() * 4);

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
            eval: EvalAccum::default(),
            history,
            prior,
        })
    }

    pub fn load_triggers(&mut self, path: &Path) -> Result<()> {
        self.triggers = TriggerDict::from_path(path)?;
        Ok(())
    }

    pub fn respond(&mut self, input: &str) -> Result<Reply> {
        let t0 = Instant::now();
        let input = input.trim();
        let tok = self.tokenize_observe(input);

        let mut q = vec![0.0f32; self.embedder.dim()];
        self.embedder.embed(input, &mut q);
        self.topic.push(&q);
        let mut topic = vec![0.0f32; self.embedder.dim()];
        self.topic.mean(&mut topic);

        let trigger_hit = self.triggers.match_one(&self.embedder, input, &self.params);

        let mut texts: Vec<String> = Vec::new();
        let mut toks: Vec<Vec<TokenId>> = Vec::new();
        let mut steps = Vec::new();
        let mut path = PathKind::Markov;
        let mut trigger_tr = None;

        if let Some((tr, responses)) = trigger_hit {
            path = PathKind::Trigger;
            trigger_tr = Some(tr);
            for r in responses {
                if r.trim().is_empty() {
                    continue;
                }
                let tk = self.tokenizer.tokenize(&mut self.intern, &r);
                texts.push(r);
                toks.push(tk.chunks);
            }
        }

        if texts.is_empty() {
            path = PathKind::Markov;
            let ctx_seed: Vec<TokenId> = self.history.iter().copied().collect();
            let parrot = tok.chunks.clone();
            let mut seen = FxHashSet::default();
            let mut attempts = 0;
            while texts.len() < self.params.n_cand && attempts < self.params.n_cand * 4 {
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
                let key = g.tokens.clone();
                if !seen.insert(key) {
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
                    steps = g.steps;
                }
                toks.push(g.tokens);
                texts.push(text);
            }
            if texts.is_empty() {
                let parrot_text = detokenize(&tok.chunk_strs);
                let shuffled = parrot_variant(&tok.chunk_strs, &mut self.rng);
                texts.push(shuffled);
                toks.push(tok.chunks.clone());
                if parrot_text != texts[0] {
                    texts.push(parrot_text);
                    toks.push(tok.chunks.clone());
                }
            }
        }

        let ranked = rank_and_pick(
            &self.embedder,
            &topic,
            &texts,
            &toks,
            &self.params,
            &mut self.rng,
        );

        let chosen_tokens = toks.get(ranked.index).cloned().unwrap_or_default();
        let chosen_text = texts
            .get(ranked.index)
            .cloned()
            .unwrap_or_else(|| "…".into());
        let sim = ranked
            .traces
            .iter()
            .find(|c| c.chosen)
            .map(|c| c.score)
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

        let trace = Trace {
            seed: self.seed,
            path,
            input: input.to_string(),
            morphemes: tok.morph_strs,
            chunks: tok.chunk_strs,
            topic_hits: self.topic.len(),
            trigger: trigger_tr,
            candidates: ranked.traces,
            chosen_rank,
            slipped: ranked.slipped,
            slip_roll: ranked.slip_roll,
            p_slip: self.params.p_slip,
            steps,
            elapsed_us: t0.elapsed().as_micros(),
            novelty_lcs,
            similarity: sim,
            band_hit,
        };

        self.eval.observe(&trace, chosen_tokens.len());
        self.commit(Role::User, input, None, None)?;
        self.commit(Role::Bot, &chosen_text, Some(ranked.slipped), Some(sim))?;
        self.store.push_utterance(&tok.chunks);
        self.store.push_utterance(&chosen_tokens);
        for &c in tok.chunks.iter().chain(chosen_tokens.iter()) {
            self.history.push_back(c);
        }
        trim_history(&mut self.history, self.params.l_max_capped() * 4);
        self.prior.push(tok.chunks);
        self.prior.push(chosen_tokens);
        self.last_trace = Some(trace.clone());

        Ok(Reply {
            text: chosen_text,
            trace,
        })
    }

    fn tokenize_observe(&mut self, text: &str) -> Tokenized {
        self.tokenizer.observe(text);
        self.tokenizer.tokenize(&mut self.intern, text)
    }

    fn commit(
        &mut self,
        role: Role,
        text: &str,
        slipped: Option<bool>,
        score: Option<f32>,
    ) -> Result<()> {
        self.log.append(Record {
            v: 1,
            t: now_ms(),
            role,
            text: text.to_string(),
            slipped,
            score,
        })
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
            tokens: self.store.len(),
            vocab: self.intern.vocab_user(),
            buf: self.store.buf.len(),
            topic_window: self.topic.len(),
        }
    }

    pub fn eval_summary(&self) -> String {
        self.eval.summary(&self.params)
    }

    pub fn params(&self) -> &Params {
        &self.params
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Rebuild tokenizer + SA from the log (source of truth).
    pub fn retokenize_from_log(&mut self) -> Result<()> {
        let texts: Vec<(Role, String)> = self
            .log
            .records
            .iter()
            .map(|r| (r.role, r.text.clone()))
            .collect();
        self.tokenizer = Tokenizer::new(&self.params);
        for (_, t) in &texts {
            self.tokenizer.observe(t);
        }
        self.intern = Interner::new();
        self.store = Store::new(self.params.merge_threshold);
        self.history.clear();
        self.prior.clear();
        self.topic = TopicTracker::new(self.params.embed_dim, self.params.k_topic);
        for (_, t) in &texts {
            let tok = self.tokenizer.tokenize(&mut self.intern, t);
            self.store.push_utterance(&tok.chunks);
            self.prior.push(tok.chunks.clone());
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
