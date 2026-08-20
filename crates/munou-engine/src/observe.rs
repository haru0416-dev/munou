//! Observation window: growth gauges from existing metrics.
//!
//! This is the product surface for 「育てられる」. It does not add a personality
//! model. Numbers come from the JSONL log, the suffix-array store, and `/eval`.

use crate::eval::EvalAccum;
use crate::explain::{PathKind, Trace};
use crate::log::{Record, Role};
use crate::params::Params;
use crate::Stats;

/// Data-driven growth stage. Labels are counts, not fake emotion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// No utterances yet.
    Empty,
    /// Log has lines but none were absorbed into the corpus.
    Logged,
    /// Something is in the store, still small.
    Sprout,
    /// Seed-scale or a short live log (tokens in the tens–hundreds).
    Growing,
    /// Large store. May be rote-heavy; that is called out separately.
    Dense,
}

impl Stage {
    pub fn label(self) -> &'static str {
        match self {
            Stage::Empty => "空",
            Stage::Logged => "記録中",
            Stage::Sprout => "芽生え",
            Stage::Growing => "育ち",
            Stage::Dense => "濃い",
        }
    }

    pub fn from_counts(utterances: usize, learned: usize, tokens: usize) -> Self {
        if utterances == 0 {
            Stage::Empty
        } else if learned == 0 {
            Stage::Logged
        } else if tokens < 40 {
            Stage::Sprout
        } else if tokens < 400 {
            Stage::Growing
        } else {
            Stage::Dense
        }
    }
}

#[derive(Debug, Clone)]
pub struct Observe {
    pub utterances: usize,
    pub learned: usize,
    pub tokens: usize,
    pub vocab: usize,
    pub buf: usize,
    pub absorb_rate: f32,
    pub eval_n: u32,
    pub band_hit_pct: f32,
    pub mean_sim: f32,
    pub rote_lcs: f32,
    pub slip_pct: f32,
    pub path_trig: u32,
    pub path_retr: u32,
    pub path_mark: u32,
    pub path_echo: u32,
    pub path_known: u32,
    pub last_path: Option<PathKind>,
    pub last_learned: Option<bool>,
    pub last_sim: Option<f32>,
    pub last_slipped: Option<bool>,
    pub last_why: Option<String>,
    pub recent_learned_bot: Vec<String>,
    pub stage: Stage,
    /// High token-LCS against prior bot utterances. Honest, not a mood.
    pub rote_lean: bool,
    pub band_lo: f32,
    pub band_hi: f32,
}

impl Observe {
    pub fn from_parts(
        stats: &Stats,
        params: &Params,
        records: &[Record],
        last: Option<&Trace>,
        eval: &EvalAccum,
    ) -> Self {
        let absorb_rate = if stats.utterances == 0 {
            0.0
        } else {
            stats.learned as f32 / stats.utterances as f32
        };

        let (eval_n, band_hit_pct, mean_sim, rote_lcs, slip_pct) = if eval.n == 0 {
            (0, 0.0, 0.0, 0.0, 0.0)
        } else {
            let n = eval.n as f32;
            let rote = if eval.lcs_len_sum == 0 {
                0.0
            } else {
                eval.lcs_sum as f32 / eval.lcs_len_sum as f32
            };
            (
                eval.n,
                100.0 * eval.band_hits as f32 / n,
                eval.sim_sum / n,
                rote,
                100.0 * eval.slip_n as f32 / n,
            )
        };

        let mut path_trig = 0u32;
        let mut path_retr = 0u32;
        let mut path_mark = 0u32;
        let mut path_echo = 0u32;
        for rec in records {
            if rec.role != Role::Bot {
                continue;
            }
            match rec.path {
                Some(PathKind::Trigger) => path_trig += 1,
                Some(PathKind::Retrieve) => path_retr += 1,
                Some(PathKind::Markov) => path_mark += 1,
                Some(PathKind::Echo) => path_echo += 1,
                None => {}
            }
        }
        let path_known = path_trig + path_retr + path_mark + path_echo;

        let last_bot = records.iter().rev().find(|r| r.role == Role::Bot);
        let last_path = last
            .map(|t| t.path)
            .or_else(|| last_bot.and_then(|r| r.path));
        let last_learned = last
            .map(|t| t.learned)
            .or_else(|| last_bot.map(|r| r.learned));
        let last_sim = last
            .map(|t| t.similarity)
            .or_else(|| last_bot.and_then(|r| r.score));
        let last_slipped = last
            .map(|t| t.slipped)
            .or_else(|| last_bot.and_then(|r| r.slipped));
        let last_why = last.map(|t| t.explain_compact()).or_else(|| {
            last_bot.map(|r| {
                format!(
                    "path={} learned={} sim={} slipped={}",
                    r.path.map(|p| p.tag()).unwrap_or("?"),
                    r.learned,
                    r.score
                        .map(|s| format!("{s:.3}"))
                        .unwrap_or_else(|| "-".into()),
                    r.slipped
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "-".into())
                )
            })
        });

        let mut recent_learned_bot = Vec::new();
        for rec in records.iter().rev() {
            if rec.role == Role::Bot && rec.learned {
                recent_learned_bot.push(rec.text.clone());
                if recent_learned_bot.len() >= 5 {
                    break;
                }
            }
        }
        recent_learned_bot.reverse();

        let stage = Stage::from_counts(stats.utterances, stats.learned, stats.tokens);
        let rote_lean = eval.n > 0 && rote_lcs >= 0.65;

        Self {
            utterances: stats.utterances,
            learned: stats.learned,
            tokens: stats.tokens,
            vocab: stats.vocab,
            buf: stats.buf,
            absorb_rate,
            eval_n,
            band_hit_pct,
            mean_sim,
            rote_lcs,
            slip_pct,
            path_trig,
            path_retr,
            path_mark,
            path_echo,
            path_known,
            last_path,
            last_learned,
            last_sim,
            last_slipped,
            last_why,
            recent_learned_bot,
            stage,
            rote_lean,
            band_lo: params.band_lo,
            band_hi: params.band_hi,
        }
    }

    pub fn vocab_frac(&self) -> f32 {
        (self.vocab as f32 / 200.0).clamp(0.0, 1.0)
    }

    pub fn novelty_frac(&self) -> f32 {
        if self.eval_n == 0 {
            0.0
        } else {
            (1.0 - self.rote_lcs).clamp(0.0, 1.0)
        }
    }

    /// One-line strip for the default chat surface (~80 cols).
    pub fn strip(&self) -> String {
        let rote = if self.eval_n == 0 {
            "-".into()
        } else {
            format!("{:.2}", self.rote_lcs)
        };
        let band = if self.eval_n == 0 {
            "-".into()
        } else {
            format!("{:.0}%", self.band_hit_pct)
        };
        let slip = if self.eval_n == 0 {
            "-".into()
        } else {
            format!("{:.0}%", self.slip_pct)
        };
        let path = self.last_path.map(|p| p.tag()).unwrap_or("-");
        format!(
            "観察 {}  吸収{} {:>3.0}%  語彙{} {:>3}  帯域{} {:>4}  暗記{} {:>4}  ズレ{} {:>3}  {}",
            self.stage.label(),
            bar(self.absorb_rate, 6),
            100.0 * self.absorb_rate,
            bar(self.vocab_frac(), 6),
            self.vocab,
            bar(
                if self.eval_n == 0 {
                    0.0
                } else {
                    self.band_hit_pct / 100.0
                },
                6
            ),
            band,
            bar(self.novelty_frac(), 6),
            rote,
            bar(
                if self.eval_n == 0 {
                    0.0
                } else {
                    self.slip_pct / 100.0
                },
                6
            ),
            slip,
            path
        )
    }

    /// Full raising panel (~80 cols).
    pub fn panel(&self) -> String {
        let mut s = String::new();
        let extra = if self.rote_lean {
            "（暗記寄り）"
        } else {
            ""
        };
        s.push_str(&format!(
            "人工無脳君 観察窓  stage={}{}\n",
            self.stage.label(),
            extra
        ));
        s.push_str(&format!(
            "発話 {}  吸収 {} ({:.0}%)  tokens={}  vocab={}  buf={}\n",
            self.utterances,
            self.learned,
            100.0 * self.absorb_rate,
            self.tokens,
            self.vocab,
            self.buf
        ));
        s.push('\n');
        s.push_str(&format!(
            "吸収 {}  {:>3.0}%\n",
            bar(self.absorb_rate, 20),
            100.0 * self.absorb_rate
        ));
        s.push_str(&format!(
            "語彙 {}  {:>3}  (満タン目安 200)\n",
            bar(self.vocab_frac(), 20),
            self.vocab
        ));
        if self.eval_n == 0 {
            s.push_str("帯域 ░░░░░░░░░░░░░░░░░░░░   —  応答スコア未記録\n");
            s.push_str("暗記 ░░░░░░░░░░░░░░░░░░░░   —  （バーは新規性。高いほど丸暗記が少ない）\n");
            s.push_str("ズレ ░░░░░░░░░░░░░░░░░░░░   —\n");
        } else {
            s.push_str(&format!(
                "帯域 {}  {:>3.0}%  [{:.2},{:.2}]  n={}  mean_sim={:.3}\n",
                bar(self.band_hit_pct / 100.0, 20),
                self.band_hit_pct,
                self.band_lo,
                self.band_hi,
                self.eval_n,
                self.mean_sim
            ));
            s.push_str(&format!(
                "暗記 {}  rote={:.2}  （バーは 1−rote。高いほど組み換え）\n",
                bar(self.novelty_frac(), 20),
                self.rote_lcs
            ));
            s.push_str(&format!(
                "ズレ {}  {:>3.0}%  slip\n",
                bar(self.slip_pct / 100.0, 20),
                self.slip_pct
            ));
        }
        s.push('\n');
        if self.path_known == 0 {
            s.push_str("経路  未記録（旧ログに path なし）\n");
        } else {
            s.push_str(&format!(
                "経路  trig={} retr={} mark={} echo={}\n",
                self.path_trig, self.path_retr, self.path_mark, self.path_echo
            ));
        }
        if let Some(why) = &self.last_why {
            s.push_str(&format!("最終  {why}\n"));
        } else {
            s.push_str("最終  （まだ応答なし）\n");
        }
        s.push_str("最近の吸収 (bot)\n");
        if self.recent_learned_bot.is_empty() {
            s.push_str("  （なし）\n");
        } else {
            for line in &self.recent_learned_bot {
                s.push_str(&format!("  ・{}\n", trunc(line, 60)));
            }
        }
        s
    }

    /// Self-contained HTML. No network, no scripts.
    pub fn html(&self) -> String {
        let extra = if self.rote_lean {
            "（暗記寄り）"
        } else {
            ""
        };
        let band_w = if self.eval_n == 0 {
            0.0
        } else {
            self.band_hit_pct
        };
        let slip_w = if self.eval_n == 0 { 0.0 } else { self.slip_pct };
        let mut recent = String::new();
        if self.recent_learned_bot.is_empty() {
            recent.push_str("<li>（なし）</li>");
        } else {
            for line in &self.recent_learned_bot {
                recent.push_str(&format!("<li>{}</li>", esc(line)));
            }
        }
        let path = if self.path_known == 0 {
            "未記録（旧ログに path なし）".into()
        } else {
            format!(
                "trig={} retr={} mark={} echo={}",
                self.path_trig, self.path_retr, self.path_mark, self.path_echo
            )
        };
        let last = self.last_why.as_deref().unwrap_or("（まだ応答なし）");
        let eval_note = if self.eval_n == 0 {
            "応答スコア未記録。シードログだけでは帯域・暗記・ズレは空。"
        } else {
            "ゲージは既存の eval 指標。感情モデルではない。"
        };
        let absorb_pct = 100.0 * self.absorb_rate;
        let absorb_w = format!("{:.1}%", absorb_pct);
        let vocab_w = format!("{:.1}%", 100.0 * self.vocab_frac());
        let band_fill = format!("{:.1}%", band_w);
        let nov_w = format!("{:.1}%", 100.0 * self.novelty_frac());
        let slip_fill = format!("{:.1}%", slip_w);
        format!(
            r##"<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>人工無脳君 観察窓</title>
<style>
body {{ font: 15px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
  max-width: 40rem; margin: 2rem auto; padding: 0 1rem;
  color: #1b1b1b; background: #f4efe6; }}
h1 {{ font-size: 1.05rem; font-weight: 700; }}
.row {{ display: flex; align-items: center; gap: .55rem; margin: .4rem 0; }}
.lbl {{ width: 3.2em; }}
.track {{ flex: 1; height: .7rem; background: #d8d0c4; border-radius: 99px; overflow: hidden; }}
.fill {{ height: 100%; background: #3c5a41; }}
.val {{ width: 7.5em; text-align: right; color: #333; }}
.note {{ color: #555; font-size: .85rem; margin-top: 1rem; }}
ul {{ padding-left: 1.2rem; }}
</style>
</head>
<body>
<h1>人工無脳君 観察窓 — {stage}{extra}</h1>
<p>発話 {utt}　吸収 {learned} ({absorb_pct:.0}%)　tokens={tokens}　vocab={vocab}　buf={buf}</p>
<div class="row"><span class="lbl">吸収</span><div class="track"><div class="fill" style="width:{absorb_w}"></div></div><span class="val">{absorb_pct:.0}%</span></div>
<div class="row"><span class="lbl">語彙</span><div class="track"><div class="fill" style="width:{vocab_w}"></div></div><span class="val">{vocab}</span></div>
<div class="row"><span class="lbl">帯域</span><div class="track"><div class="fill" style="width:{band_fill}"></div></div><span class="val">{band_txt}</span></div>
<div class="row"><span class="lbl">暗記</span><div class="track"><div class="fill" style="width:{nov_w}"></div></div><span class="val">{rote_txt}</span></div>
<div class="row"><span class="lbl">ズレ</span><div class="track"><div class="fill" style="width:{slip_fill}"></div></div><span class="val">{slip_txt}</span></div>
<p>経路 {path}</p>
<p>最終 {last}</p>
<p>最近の吸収 (bot)</p>
<ul>{recent}</ul>
<p class="note">{eval_note} ファイルはローカル生成。サーバもアダプタも無い。</p>
</body>
</html>
"##,
            stage = esc(self.stage.label()),
            extra = extra,
            utt = self.utterances,
            learned = self.learned,
            absorb_pct = absorb_pct,
            absorb_w = absorb_w,
            tokens = self.tokens,
            vocab = self.vocab,
            buf = self.buf,
            vocab_w = vocab_w,
            band_fill = band_fill,
            band_txt = if self.eval_n == 0 {
                "—".into()
            } else {
                format!("{:.0}%", self.band_hit_pct)
            },
            nov_w = nov_w,
            rote_txt = if self.eval_n == 0 {
                "—".into()
            } else {
                format!("rote={:.2}", self.rote_lcs)
            },
            slip_fill = slip_fill,
            slip_txt = if self.eval_n == 0 {
                "—".into()
            } else {
                format!("{:.0}%", self.slip_pct)
            },
            path = esc(&path),
            last = esc(last),
            recent = recent,
            eval_note = eval_note,
        )
    }
}

pub fn bar(frac: f32, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let filled = ((frac.clamp(0.0, 1.0) * width as f32).round() as usize).min(width);
    let mut s = String::with_capacity(width * 3);
    for i in 0..width {
        s.push(if i < filled { '█' } else { '░' });
    }
    s
}

fn trunc(s: &str, max_chars: usize) -> String {
    let n = s.chars().count();
    if n <= max_chars {
        return s.to_string();
    }
    let t: String = s.chars().take(max_chars.saturating_sub(1)).collect();
    format!("{t}…")
}

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => o.push_str("&amp;"),
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            '"' => o.push_str("&quot;"),
            '\'' => o.push_str("&#39;"),
            c => o.push(c),
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::EvalAccum;
    use crate::params::Params;

    fn empty_stats() -> Stats {
        Stats {
            utterances: 0,
            learned: 0,
            tokens: 0,
            vocab: 0,
            buf: 0,
            topic_window: 0,
        }
    }

    #[test]
    fn empty_stage_and_gauges() {
        let o = Observe::from_parts(
            &empty_stats(),
            &Params::default(),
            &[],
            None,
            &EvalAccum::default(),
        );
        assert_eq!(o.stage, Stage::Empty);
        assert_eq!(o.stage.label(), "空");
        assert_eq!(o.absorb_rate, 0.0);
        let p = o.panel();
        assert!(p.contains("空"), "{p}");
        assert!(!p.is_empty());
        assert!(o.strip().contains("観察"));
    }

    #[test]
    fn logged_unlearned_is_not_sprout() {
        let st = Stats {
            utterances: 4,
            learned: 0,
            tokens: 0,
            vocab: 0,
            buf: 0,
            topic_window: 0,
        };
        let o = Observe::from_parts(&st, &Params::default(), &[], None, &EvalAccum::default());
        assert_eq!(o.stage, Stage::Logged);
        assert_eq!(o.stage.label(), "記録中");
    }

    #[test]
    fn html_escapes_recent_lines() {
        let st = Stats {
            utterances: 2,
            learned: 2,
            tokens: 8,
            vocab: 4,
            buf: 0,
            topic_window: 1,
        };
        let rec = Record {
            v: 1,
            t: 0,
            role: Role::Bot,
            text: "<script>x</script>".into(),
            slipped: None,
            score: None,
            learned: true,
            path: Some(PathKind::Echo),
            novelty_lcs: None,
            n_tok: None,
        };
        let o = Observe::from_parts(&st, &Params::default(), &[rec], None, &EvalAccum::default());
        let h = o.html();
        assert!(h.contains("&lt;script&gt;"), "{h}");
        assert!(!h.contains("<script>x"), "{h}");
        assert!(h.contains("charset=\"utf-8\""));
    }

    #[test]
    fn bar_bounds() {
        assert_eq!(bar(0.0, 4).chars().filter(|&c| c == '█').count(), 0);
        assert_eq!(bar(1.0, 4).chars().filter(|&c| c == '█').count(), 4);
        assert_eq!(bar(0.5, 4).chars().filter(|&c| c == '█').count(), 2);
    }
}
