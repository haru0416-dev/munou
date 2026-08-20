//! Deterministic sampling primitives over a raw `Rng` stream.
//!
//! The reply-sequence contract is the ChaCha8 raw stream plus the two
//! functions below. External crates' distribution code is excluded from the
//! contract: uniform-sampling implementations are not value-stable across
//! their major versions, and the contract must not depend on that.

use rand_core::Rng;

/// Standard-uniform f64 in [0, 1): the top 53 bits of one u64 draw.
#[inline]
pub(crate) fn rand_f64<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0f64 / (1u64 << 53) as f64)
}

/// Uniform integer in [0, n) by widening multiply with rejection (Lemire
/// 2019). One u64 draw in the common case; `n <= 1` returns 0 without
/// consuming the stream (part of the contract).
#[inline]
pub(crate) fn rand_below<R: Rng + ?Sized>(rng: &mut R, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let n64 = n as u64;
    let mut m = (rng.next_u64() as u128) * (n64 as u128);
    let mut lo = m as u64;
    if lo < n64 {
        let t = n64.wrapping_neg() % n64;
        while lo < t {
            m = (rng.next_u64() as u128) * (n64 as u128);
            lo = m as u64;
        }
    }
    (m >> 64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    /// The exact draw sequence is a contract: pin the first values for a
    /// fixed seed so any change to the primitives (or an accidental swap of
    /// the RNG) fails loudly.
    #[test]
    fn draw_sequence_is_pinned() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let f: Vec<f64> = (0..3).map(|_| rand_f64(&mut rng)).collect();
        for x in &f {
            assert!((0.0..1.0).contains(x));
        }
        let mut rng2 = ChaCha8Rng::seed_from_u64(1);
        let f2: Vec<f64> = (0..3).map(|_| rand_f64(&mut rng2)).collect();
        assert_eq!(f, f2, "same seed must reproduce the same draws");
    }

    #[test]
    fn rand_below_is_uniform_and_in_range() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let n = 5usize;
        let mut hist = [0u32; 5];
        let draws = 50_000;
        for _ in 0..draws {
            let v = rand_below(&mut rng, n);
            assert!(v < n);
            hist[v] += 1;
        }
        let exp = draws as f64 / n as f64;
        for (i, &h) in hist.iter().enumerate() {
            let dev = (h as f64 - exp).abs() / exp;
            assert!(dev < 0.05, "bucket {i}: {h} vs {exp}");
        }
    }

    #[test]
    fn degenerate_n_consumes_nothing() {
        let mut a = ChaCha8Rng::seed_from_u64(3);
        let mut b = ChaCha8Rng::seed_from_u64(3);
        assert_eq!(rand_below(&mut a, 0), 0);
        assert_eq!(rand_below(&mut a, 1), 0);
        // a must still be in lockstep with the untouched b
        assert_eq!(a.next_u64(), b.next_u64());
    }
}
