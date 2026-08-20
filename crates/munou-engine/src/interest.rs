//! 関心 — dual-timescale chunk weights (熱 short, 根 long half-life) on the
//! log-position clock, never wall time: replay rebuilds identical values.
//!
//! 聞きかじり: chunks in fewer than `hearsay_min` distinct utterances carry
//! no weight and cannot anchor (a one-off typo must not anchor a reply),
//! except on 口をつく turns (`hearsay_release`).

use rustc_hash::FxHashMap;

use crate::ids::{is_special, TokenId};

/// Half-life of 熱, in absorbed tokens.
pub(crate) const HALF_FAST: f64 = 300.0;
/// Half-life of 根, in absorbed tokens.
pub(crate) const HALF_SLOW: f64 = 10_000.0;
/// 熱 contributes at most this much to the score.
const HEAT_CAP: f64 = 6.0;
/// 根 contributes at most this much (after log10 compression).
const ROOT_CAP: f64 = 4.0;
/// Raw heat may exceed HEAT_CAP: repeated mentions then extend how long the
/// term stays saturated after they stop.
const HEAT_RAW_CAP: f64 = 2.0 * HEAT_CAP;

#[derive(Debug, Clone, Default)]
struct Entry {
    heat: f64,
    root: f64,
    /// Clock value at the last touch (lazy decay).
    last_g: u64,
    /// Distinct learned utterances this chunk appeared in.
    utts: u32,
}

/// (id, heat, root, last_g, utts) per entry.
pub(crate) type InterestSnap = Vec<(TokenId, f64, f64, u64, u32)>;

/// Per-chunk 関心 ledger. Rebuilt from the log on open; fed by `absorb` live.
#[derive(Debug, Default)]
pub(crate) struct InterestLedger {
    entries: FxHashMap<TokenId, Entry>,
    /// Log-position clock: absorbed tokens so far.
    g: u64,
}

impl InterestLedger {
    /// Absorb one learned utterance: each distinct content chunk counts once
    /// per utterance, then the clock advances by the utterance length.
    pub fn learn(&mut self, chunks: &[TokenId], is_punct: impl Fn(TokenId) -> bool) {
        let mut seen = rustc_hash::FxHashSet::default();
        for &id in chunks {
            if is_special(id) || is_punct(id) || !seen.insert(id) {
                continue;
            }
            let e = self.entries.entry(id).or_default();
            let d = self.g.saturating_sub(e.last_g) as f64;
            e.heat = (e.heat * 2f64.powf(-d / HALF_FAST) + 1.0).min(HEAT_RAW_CAP);
            e.root = e.root * 2f64.powf(-d / HALF_SLOW) + 1.0;
            e.last_g = self.g;
            e.utts = e.utts.saturating_add(1);
        }
        self.g += chunks.len() as u64;
    }

    /// Current 関心 in [0, 1], or `None` for unknown / hearsay chunks.
    pub fn score(&self, id: TokenId, hearsay_min: u32) -> Option<f32> {
        let e = self.entries.get(&id)?;
        if e.utts < hearsay_min {
            return None;
        }
        let d = self.g.saturating_sub(e.last_g) as f64;
        let heat = (e.heat * 2f64.powf(-d / HALF_FAST)).min(HEAT_CAP);
        let root = (2.0 * (1.0 + e.root * 2f64.powf(-d / HALF_SLOW)).log10()).min(ROOT_CAP);
        Some(((heat + root) / (HEAT_CAP + ROOT_CAP)) as f32)
    }

    /// 聞きかじり: not yet established vocabulary (unknown counts too).
    pub fn is_hearsay(&self, id: TokenId, hearsay_min: u32) -> bool {
        self.entries
            .get(&id)
            .map(|e| e.utts < hearsay_min)
            .unwrap_or(true)
    }

    /// Snapshot payload. Map order is irrelevant: every consumer either
    /// looks up by id or re-sorts by surface string.
    pub(crate) fn snap_dump(&self) -> (u64, InterestSnap) {
        (
            self.g,
            self.entries
                .iter()
                .map(|(id, e)| (*id, e.heat, e.root, e.last_g, e.utts))
                .collect(),
        )
    }

    pub(crate) fn from_snap(g: u64, entries: InterestSnap) -> Self {
        let mut m = FxHashMap::default();
        for (id, heat, root, last_g, utts) in entries {
            m.insert(
                id,
                Entry {
                    heat,
                    root,
                    last_g,
                    utts,
                },
            );
        }
        Self { entries: m, g }
    }

    /// Established chunk ids (non-hearsay), for the care-word draw and あゆみ.
    pub fn established(&self, hearsay_min: u32) -> Vec<TokenId> {
        self.entries
            .iter()
            .filter(|(_, e)| e.utts >= hearsay_min)
            .map(|(id, _)| *id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_punct(_: TokenId) -> bool {
        false
    }

    #[test]
    fn heat_halves_after_half_life() {
        let mut l = InterestLedger::default();
        l.learn(&[70], no_punct);
        let s0 = l.score(70, 1).unwrap();
        // Advance the clock by one fast half-life without touching the chunk.
        let filler: Vec<TokenId> = vec![80; HALF_FAST as usize];
        l.learn(&filler, no_punct);
        let s1 = l.score(70, 1).unwrap();
        assert!(s1 < s0, "interest must decay: {s0} -> {s1}");
        // 根 decays ~40x slower, so the drop is dominated by 熱 halving.
        assert!(s1 > s0 * 0.3, "root must survive the fast half-life");
    }

    #[test]
    fn hearsay_gate_opens_at_min_utterances() {
        let mut l = InterestLedger::default();
        l.learn(&[70], no_punct);
        assert!(l.is_hearsay(70, 2));
        assert_eq!(l.score(70, 2), None);
        l.learn(&[70], no_punct);
        assert!(!l.is_hearsay(70, 2));
        assert!(l.score(70, 2).is_some());
        assert!(l.is_hearsay(99, 2), "unknown chunks are hearsay");
    }

    #[test]
    fn repeats_inside_one_utterance_count_once() {
        let mut a = InterestLedger::default();
        a.learn(&[70, 70, 70], no_punct);
        assert!(
            a.is_hearsay(70, 2),
            "そうそうそう must not fast-track a word"
        );
    }

    #[test]
    fn score_is_bounded() {
        let mut l = InterestLedger::default();
        for _ in 0..200 {
            l.learn(&[70], no_punct);
        }
        let s = l.score(70, 1).unwrap();
        assert!((0.0..=1.0).contains(&s), "score out of range: {s}");
    }
}
