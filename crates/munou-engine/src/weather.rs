//! 日和 — the day's condition, a deterministic function of (seed, day).
//!
//! The day comes from the **previous log record's timestamp**, never from the
//! wall clock at scoring time, so a reply stays a pure function of
//! (log, seed, input): replaying the same log through the same seed gives the
//! same weather. The day is the UTC day number (t_ms / 86_400_000) — no
//! timezone dependency, same result on any machine.
//!
//! 特徴のある日和は2日続けない: 素の抽選で前日と同じ特殊日和を引いたら
//! 「なぎ」に落とす。14% 程度の日和でも抽選だけだと2〜3日並ぶことは普通に
//! 起きて、観察する側には「ずっと同じ」に見える（知覚上のストリークは
//! バグに見える）。前日も同じ式で引けるので決定論は保たれる。

/// One day's condition: multipliers over the experience dials.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Weather {
    pub name: &'static str,
    /// × on `p_slip`.
    pub slip: f64,
    /// × on `interject_rate` (合いの手の出やすさ).
    pub interject: f64,
    /// × on `hearsay_release` (口をつく確率).
    pub release: f64,
    /// × on `care_bonus` (気になる語の加点).
    pub care: f32,
}

pub(crate) const CALM: Weather = Weather {
    name: "なぎ",
    slip: 1.0,
    interject: 1.0,
    release: 1.0,
    care: 1.0,
};

/// (weather, lottery weight). なぎ must stay first — the anti-streak rule
/// falls back to it.
const TABLE: [(Weather, u32); 4] = [
    (CALM, 49),
    (
        Weather {
            name: "はずみ",
            slip: 1.2,
            interject: 1.6,
            release: 1.0,
            care: 1.0,
        },
        17,
    ),
    (
        Weather {
            name: "しめり",
            slip: 0.7,
            interject: 0.4,
            release: 1.0,
            care: 0.5,
        },
        17,
    ),
    (
        Weather {
            name: "きまぐれ",
            slip: 1.8,
            interject: 1.0,
            release: 2.5,
            care: 1.5,
        },
        17,
    ),
];

fn total_weight() -> u64 {
    TABLE.iter().map(|(_, w)| *w as u64).sum()
}

/// SplitMix64 finalizer — a fixed, published mix. Not a stream: one value per
/// (seed, day), so it neither touches nor races the ChaCha8 reply stream.
pub(crate) fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

fn raw_index(seed: u64, day: u64) -> usize {
    let h = mix(seed ^ mix(day));
    let mut r = h % total_weight();
    for (i, (_, w)) in TABLE.iter().enumerate() {
        if r < *w as u64 {
            return i;
        }
        r -= *w as u64;
    }
    0
}

/// The day's weather, streak-broken: a special weather never repeats on two
/// consecutive days.
pub(crate) fn day_weather(seed: u64, day: u64) -> &'static Weather {
    let i = raw_index(seed, day);
    if i != 0 && day > 0 && raw_index(seed, day - 1) == i {
        return &TABLE[0].0;
    }
    &TABLE[i].0
}

/// UTC day number of a millisecond timestamp.
pub(crate) fn day_of_ms(t_ms: u64) -> u64 {
    t_ms / 86_400_000
}

/// Deterministic pick of the day's 気になる語 among `len` sorted candidates.
pub(crate) fn care_index(seed: u64, day: u64, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    (mix(seed ^ mix(day ^ 0xCA9E_0000)) % len as u64) as usize
}

/// Effective dials for the turn: weather × よそよそしさ (post-absence damp:
/// fewer beats, no care word, half the slip until the distance wears off).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Gains {
    pub slip: f64,
    pub interject: f64,
    pub release: f64,
    pub care: f32,
}

pub(crate) fn effective(w: &Weather, aloof: bool) -> Gains {
    let mut g = Gains {
        slip: w.slip,
        interject: w.interject,
        release: w.release,
        care: w.care,
    };
    if aloof {
        g.slip *= 0.5;
        g.interject = 0.0;
        g.care = 0.0;
    }
    g
}

/// Civil date (y, m, d) from a UTC day number (Howard Hinnant's algorithm).
pub(crate) fn civil_from_day(day: u64) -> (i64, u32, u32) {
    let z = day as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_per_seed_and_day() {
        for day in 0..64 {
            assert_eq!(day_weather(9, day).name, day_weather(9, day).name);
        }
    }

    #[test]
    fn special_weather_never_two_days_running() {
        for seed in 0..8u64 {
            let mut prev = "";
            for day in 0..3000u64 {
                let w = day_weather(seed, day).name;
                if w != "なぎ" {
                    assert_ne!(w, prev, "seed={seed} day={day}");
                }
                prev = w;
            }
        }
    }

    #[test]
    fn all_weathers_occur() {
        let mut seen = std::collections::HashSet::new();
        for day in 0..2000u64 {
            seen.insert(day_weather(3, day).name);
        }
        assert_eq!(seen.len(), TABLE.len(), "{seen:?}");
    }

    #[test]
    fn civil_dates() {
        assert_eq!(civil_from_day(0), (1970, 1, 1));
        assert_eq!(civil_from_day(19_723), (2024, 1, 1)); // leap-adjacent spot check
        assert_eq!(civil_from_day(20_684), (2026, 8, 19));
        assert_eq!(day_of_ms(20_684 * 86_400_000 + 123), 20_684);
    }

    #[test]
    fn aloof_mutes_beats_and_care() {
        let g = effective(&CALM, true);
        assert_eq!(g.interject, 0.0);
        assert_eq!(g.care, 0.0);
        assert!(g.slip < CALM.slip);
    }
}
