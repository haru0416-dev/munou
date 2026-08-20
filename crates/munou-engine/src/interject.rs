//! 合いの手 — 返答の前に出す、学習済みの短い一行。
//!
//! 固定テーブルは持たない。自ログの学習済み短発話（≤ `MAX_CHARS` 文字）を
//! 頻度つきで収穫し、その分布から引く（§2 閉世界: 素材は自ログのみ）。
//!
//! Constraints: display only — a verbatim copy of an already-learned line,
//! neither logged nor re-absorbed, so replay is unchanged. Exempt from the
//! self-repetition penalty, like triggers.

use rand_core::Rng;

use crate::rng::rand_below;

/// 合いの手として収穫する最大文字数。
const MAX_CHARS: usize = 8;
/// この数の distinct 発話が集まるまでは黙っている。
pub(crate) const MIN_DISTINCT: usize = 3;

#[derive(Debug, Default)]
pub(crate) struct InterjectBank {
    /// (text, count) in first-seen log order — the order is part of the
    /// deterministic pick and is identical live and on replay.
    items: Vec<(String, u64)>,
    index: rustc_hash::FxHashMap<String, usize>,
    total: u64,
}

impl InterjectBank {
    /// Harvest one learned utterance (either role). Long lines are ignored.
    pub fn learn(&mut self, text: &str) {
        let t = text.trim();
        let n = t.chars().count();
        if n == 0 || n > MAX_CHARS {
            return;
        }
        if t.chars()
            .all(|c| !c.is_alphanumeric() && !is_kana_or_cjk(c))
        {
            return; // punctuation-only lines are excluded
        }
        match self.index.get(t) {
            Some(&i) => self.items[i].1 += 1,
            None => {
                self.index.insert(t.to_string(), self.items.len());
                self.items.push((t.to_string(), 1));
            }
        }
        self.total += 1;
    }

    pub fn distinct(&self) -> usize {
        self.items.len()
    }

    /// Frequency-weighted draw. Returns `None` when the bank is too small or
    /// the draw lands on `exclude` (no redraw: RNG consumption per call is
    /// fixed at one draw).
    pub fn pick<R: Rng + ?Sized>(&self, rng: &mut R, exclude: &str) -> Option<String> {
        if self.distinct() < MIN_DISTINCT || self.total == 0 {
            return None;
        }
        let mut r = rand_below(rng, self.total as usize) as u64;
        for (text, c) in &self.items {
            if r < *c {
                if text == exclude {
                    return None;
                }
                return Some(text.clone());
            }
            r -= c;
        }
        None
    }

    /// Snapshot for tests and parity checks.
    #[cfg(test)]
    pub fn entries(&self) -> Vec<(String, u64)> {
        self.items.clone()
    }
}

fn is_kana_or_cjk(c: char) -> bool {
    matches!(c,
        '\u{3040}'..='\u{30ff}' | '\u{3400}'..='\u{9fff}' | '\u{ff66}'..='\u{ff9d}')
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    #[test]
    fn harvests_short_lines_with_counts() {
        let mut b = InterjectBank::default();
        for t in [
            "はい",
            "はい",
            "うん",
            "そうだね",
            "今日はとてもいい天気ですね",
        ] {
            b.learn(t);
        }
        assert_eq!(b.distinct(), 3, "long line must be ignored");
        assert_eq!(b.entries()[0], ("はい".into(), 2));
    }

    #[test]
    fn punct_only_lines_are_ignored() {
        let mut b = InterjectBank::default();
        b.learn("……");
        b.learn("！？");
        assert_eq!(b.distinct(), 0);
    }

    #[test]
    fn pick_is_deterministic_and_respects_weights() {
        let mut b = InterjectBank::default();
        for _ in 0..20 {
            b.learn("はい");
        }
        b.learn("うん");
        b.learn("おお");
        let mut r1 = ChaCha8Rng::seed_from_u64(5);
        let mut r2 = ChaCha8Rng::seed_from_u64(5);
        let picks1: Vec<_> = (0..10).map(|_| b.pick(&mut r1, "")).collect();
        let picks2: Vec<_> = (0..10).map(|_| b.pick(&mut r2, "")).collect();
        assert_eq!(picks1, picks2);
        let hai = picks1
            .iter()
            .filter(|p| p.as_deref() == Some("はい"))
            .count();
        assert!(hai >= 6, "frequency weight must dominate: {picks1:?}");
    }

    #[test]
    fn small_bank_stays_silent() {
        let mut b = InterjectBank::default();
        b.learn("はい");
        b.learn("うん");
        let mut r = ChaCha8Rng::seed_from_u64(1);
        assert_eq!(b.pick(&mut r, ""), None);
    }
}
