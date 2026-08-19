//! SA-IS suffix array over a `u32` alphabet (Nong, Zhang, Chan 2009).
//!
//! The token stream may contain `0` (EOS) many times, so each symbol is mapped
//! `t → t+1` and a unique sentinel `0` is appended before construction.

use crate::ids::TokenId;

pub fn suffix_array(text: &[TokenId]) -> Vec<u32> {
    let n = text.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }
    let mut s = Vec::with_capacity(n + 1);
    let mut max_sym = 0u32;
    for &t in text {
        let v = t.saturating_add(1);
        if v > max_sym {
            max_sym = v;
        }
        s.push(v);
    }
    s.push(0);
    let k = (max_sym as usize) + 1;
    let sa = sais(&s, k);
    sa.into_iter()
        .filter(|&i| i as usize != n)
        .map(|i| i as u32)
        .collect()
}

/// Binary-search the half-open SA range whose suffixes start with `pat`.
pub fn sa_range(text: &[TokenId], sa: &[u32], pat: &[TokenId]) -> Option<(usize, usize)> {
    if pat.is_empty() || sa.is_empty() {
        return None;
    }
    let lo = bound(text, sa, pat, true);
    let hi = bound(text, sa, pat, false);
    if lo < hi {
        Some((lo, hi))
    } else {
        None
    }
}

fn cmp_prefix(text: &[TokenId], i: usize, pat: &[TokenId]) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    for (k, &p) in pat.iter().enumerate() {
        match text.get(i + k) {
            None => return Less,
            Some(&c) if c < p => return Less,
            Some(&c) if c > p => return Greater,
            Some(_) => {}
        }
    }
    Equal
}

fn bound(text: &[TokenId], sa: &[u32], pat: &[TokenId], lower: bool) -> usize {
    let mut lo = 0usize;
    let mut hi = sa.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let ord = cmp_prefix(text, sa[mid] as usize, pat);
        let take_left = if lower {
            ord != std::cmp::Ordering::Less
        } else {
            ord == std::cmp::Ordering::Greater
        };
        if take_left {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

fn is_lms(stype: &[bool], i: usize) -> bool {
    i > 0 && stype[i] && !stype[i - 1]
}

fn sais(s: &[u32], k: usize) -> Vec<i32> {
    let n = s.len();
    let mut stype = vec![false; n];
    stype[n - 1] = true;
    for i in (0..n - 1).rev() {
        stype[i] = s[i] < s[i + 1] || (s[i] == s[i + 1] && stype[i + 1]);
    }

    let lms_pos: Vec<usize> = (0..n).filter(|&i| is_lms(&stype, i)).collect();
    let sa = induce(s, &stype, k, &lms_pos);

    if lms_pos.is_empty() {
        return sa;
    }

    let mut lms_sorted = Vec::with_capacity(lms_pos.len());
    for &p in &sa {
        let p = p as usize;
        if is_lms(&stype, p) {
            lms_sorted.push(p);
        }
    }

    let mut names = vec![-1i32; n];
    let mut name = 0i32;
    names[lms_sorted[0]] = 0;
    for w in lms_sorted.windows(2) {
        if !lms_substr_eq(s, &stype, w[0], w[1]) {
            name += 1;
        }
        names[w[1]] = name;
    }

    let s1: Vec<u32> = lms_pos.iter().map(|&i| names[i] as u32).collect();
    let n1 = s1.len();
    let sa1 = if (name as usize) + 1 == n1 {
        let mut sa1 = vec![0i32; n1];
        for (i, &c) in s1.iter().enumerate() {
            sa1[c as usize] = i as i32;
        }
        sa1
    } else {
        sais(&s1, (name as usize) + 1)
    };

    let ordered: Vec<usize> = sa1.iter().map(|&i| lms_pos[i as usize]).collect();
    induce(s, &stype, k, &ordered)
}

fn lms_substr_eq(s: &[u32], stype: &[bool], mut a: usize, mut b: usize) -> bool {
    if a == b {
        return true;
    }
    let n = s.len();
    let mut first = true;
    loop {
        if a >= n || b >= n {
            return a >= n && b >= n;
        }
        if s[a] != s[b] || stype[a] != stype[b] {
            return false;
        }
        let a_lms = is_lms(stype, a);
        let b_lms = is_lms(stype, b);
        if !first && a_lms && b_lms {
            return true;
        }
        if a_lms != b_lms {
            return false;
        }
        first = false;
        a += 1;
        b += 1;
    }
}

fn counts(s: &[u32], k: usize) -> Vec<i32> {
    let mut c = vec![0i32; k];
    for &x in s {
        c[x as usize] += 1;
    }
    c
}

fn buckets(cnt: &[i32], end: bool) -> Vec<i32> {
    let mut bkt = vec![0i32; cnt.len()];
    let mut sum = 0i32;
    for i in 0..cnt.len() {
        sum += cnt[i];
        bkt[i] = if end { sum } else { sum - cnt[i] };
    }
    bkt
}

fn induce(s: &[u32], stype: &[bool], k: usize, lms_in_sa_order_or_not: &[usize]) -> Vec<i32> {
    let n = s.len();
    let cnt = counts(s, k);
    let mut sa = vec![-1i32; n];

    let mut bkt = buckets(&cnt, true);
    for &i in lms_in_sa_order_or_not.iter().rev() {
        let c = s[i] as usize;
        bkt[c] -= 1;
        sa[bkt[c] as usize] = i as i32;
    }

    let mut bkt = buckets(&cnt, false);
    for i in 0..n {
        let v = sa[i];
        if v <= 0 {
            continue;
        }
        let j = (v as usize) - 1;
        if !stype[j] {
            let c = s[j] as usize;
            sa[bkt[c] as usize] = j as i32;
            bkt[c] += 1;
        }
    }

    let mut bkt = buckets(&cnt, true);
    for i in (0..n).rev() {
        let v = sa[i];
        if v <= 0 {
            continue;
        }
        let j = (v as usize) - 1;
        if stype[j] {
            let c = s[j] as usize;
            bkt[c] -= 1;
            sa[bkt[c] as usize] = j as i32;
        }
    }
    sa
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(text: &[TokenId]) -> Vec<u32> {
        let n = text.len();
        let mut sa: Vec<u32> = (0..n as u32).collect();
        sa.sort_by(|&i, &j| text[i as usize..].cmp(&text[j as usize..]));
        sa
    }

    #[test]
    fn mississippi() {
        let s: Vec<TokenId> = b"mississippi".iter().map(|&c| c as u32).collect();
        assert_eq!(suffix_array(&s), naive(&s));
    }

    #[test]
    fn banana() {
        let s: Vec<TokenId> = b"banana".iter().map(|&c| c as u32).collect();
        assert_eq!(suffix_array(&s), naive(&s));
    }

    #[test]
    fn repeated_zeros() {
        let s = vec![0u32, 1, 0, 1, 0, 2, 0];
        assert_eq!(suffix_array(&s), naive(&s));
    }

    #[test]
    fn random_matches_naive() {
        let mut x = 0x1234_5678u64;
        for n in [1usize, 2, 3, 8, 17, 64, 200, 512] {
            let mut s = Vec::with_capacity(n);
            for _ in 0..n {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                s.push(((x >> 33) as u32) % 7);
            }
            assert_eq!(suffix_array(&s), naive(&s), "n={n}");
        }
    }

    #[test]
    fn range_finds_pattern() {
        let s: Vec<TokenId> = b"banana".iter().map(|&c| c as u32).collect();
        let sa = suffix_array(&s);
        let pat = vec![b'a' as u32, b'n' as u32];
        let (lo, hi) = sa_range(&s, &sa, &pat).unwrap();
        assert!(hi - lo >= 2);
        for &i in &sa[lo..hi] {
            let i = i as usize;
            assert_eq!(&s[i..i + 2], pat.as_slice());
        }
    }
}
