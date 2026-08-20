//! 節目 — growth marks from digest crossings; counts and days, not fake
//! emotion. A crossing is a property of adjacent records, so replay never
//! re-fires a mark and reopening cannot lose one.
//!
//! よそよそしさ: a speech gap ≥ `ALOOF_GAP_DAYS` days damps the dials for the
//! next `ALOOF_SPEECH` speech records (`weather::effective`).

use crate::observe::LogDigest;
use crate::weather::day_of_ms;

/// 吸収数の節目。
pub(crate) const LEARNED_MARKS: [usize; 5] = [10, 100, 1_000, 10_000, 100_000];
/// 初回記録からの日数の節目。
pub(crate) const DAY_MARKS: [u64; 6] = [7, 30, 100, 365, 730, 1095];
/// この日数以上あくと、よそよそしくなる。
pub(crate) const ALOOF_GAP_DAYS: u64 = 7;
/// よそよそしさが解けるまでの発話レコード数。
pub(crate) const ALOOF_SPEECH: u32 = 20;

/// Marks crossed by the records just ingested. `pre_*` are the digest values
/// captured before the append. At most a couple of short lines.
pub(crate) fn lines(
    pre_learned: usize,
    pre_last_t: Option<u64>,
    pre_aloof: u32,
    d: &LogDigest,
) -> Vec<String> {
    let mut out = Vec::new();
    // Reunion first: after a long absence it is the line that matters.
    if pre_aloof == 0 && d.aloof_left > 0 && d.last_gap_days >= ALOOF_GAP_DAYS {
        out.push(format!("{}日ぶり", d.last_gap_days));
    }
    // A single turn can cross several day marks (long gap): report only the
    // largest — 「730日目」, not 「7日目」.
    if let (Some(first), Some(now)) = (d.first_speech_t, d.last_speech_t) {
        let first_day = day_of_ms(first);
        let now_age = day_of_ms(now).saturating_sub(first_day);
        if let Some(pre_t) = pre_last_t {
            let pre_age = day_of_ms(pre_t).saturating_sub(first_day);
            if let Some(dm) = DAY_MARKS
                .iter()
                .filter(|dm| pre_age < **dm && now_age >= **dm)
                .max()
            {
                out.push(format!("節目 {dm}日目"));
            }
        }
    }
    for m in LEARNED_MARKS {
        if pre_learned < m && d.learned >= m {
            out.push(format!("節目 吸収{m}"));
        }
    }
    out
}

/// Achieved marks for あゆみ: (learned marks reached, day marks reached).
pub(crate) fn achieved(learned: usize, age_days: u64) -> (Vec<usize>, Vec<u64>) {
    (
        LEARNED_MARKS
            .iter()
            .copied()
            .filter(|m| learned >= *m)
            .collect(),
        DAY_MARKS
            .iter()
            .copied()
            .filter(|m| age_days >= *m)
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::log::{Record, Role};

    fn rec(role: Role, t: u64, learned: bool) -> Record {
        Record {
            v: 1,
            t,
            role,
            text: "x".into(),
            slipped: None,
            score: None,
            learned,
            path: None,
            novelty_lcs: None,
            n_tok: None,
            rng_word_pos: None,
        }
    }

    #[test]
    fn learned_crossing_fires_once() {
        let mut d = LogDigest::default();
        for i in 0..9 {
            d.ingest(&rec(Role::User, 1000 + i, true));
        }
        let pre = (d.learned, d.last_speech_t, d.aloof_left);
        d.ingest(&rec(Role::User, 2000, true));
        let l = lines(pre.0, pre.1, pre.2, &d);
        assert_eq!(l, vec!["節目 吸収10".to_string()]);
        // The next record must not re-fire it.
        let pre = (d.learned, d.last_speech_t, d.aloof_left);
        d.ingest(&rec(Role::User, 3000, true));
        assert!(lines(pre.0, pre.1, pre.2, &d).is_empty());
    }

    #[test]
    fn day_mark_fires_on_crossing_turn() {
        let day = 86_400_000u64;
        let mut d = LogDigest::default();
        d.ingest(&rec(Role::User, 10, false));
        d.ingest(&rec(Role::User, 3 * day, false));
        let pre = (d.learned, d.last_speech_t, d.aloof_left);
        d.ingest(&rec(Role::User, 8 * day, false));
        let l = lines(pre.0, pre.1, pre.2, &d);
        assert!(l.iter().any(|s| s == "節目 7日目"), "{l:?}");
    }

    #[test]
    fn absence_sets_aloof_and_reports_gap() {
        let day = 86_400_000u64;
        let mut d = LogDigest::default();
        d.ingest(&rec(Role::User, 10, false));
        let pre = (d.learned, d.last_speech_t, d.aloof_left);
        d.ingest(&rec(Role::User, 9 * day, false));
        assert!(d.aloof_left > 0);
        let l = lines(pre.0, pre.1, pre.2, &d);
        assert!(l.iter().any(|s| s.ends_with("日ぶり")), "{l:?}");
        // よそよそしさは発話レコードで漸減する。
        for i in 0..ALOOF_SPEECH {
            d.ingest(&rec(Role::Bot, 9 * day + 1 + i as u64, false));
        }
        assert_eq!(d.aloof_left, 0);
    }
}
