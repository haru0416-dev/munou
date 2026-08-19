//! Closed feature-hash embedder. Used **only** for selection / topic tracking,
//! never for generation. A pretrained model can replace this type later via
//! the same `Embedder` surface without touching the Markov path.

use crate::params::Params;

pub trait Embedder {
    fn dim(&self) -> usize;
    fn embed(&self, text: &str, out: &mut [f32]);
}

/// Signed hashing trick over character n-grams (n = 1..=4).
#[derive(Debug, Clone)]
pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim: dim.max(8) }
    }

    pub fn from_params(params: &Params) -> Self {
        Self::new(params.embed_dim)
    }
}

impl Embedder for HashEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str, out: &mut [f32]) {
        let d = self.dim.min(out.len());
        out[..d].fill(0.0);
        let chars: Vec<char> = text.chars().collect();
        if chars.is_empty() {
            return;
        }
        for n in 1..=4.min(chars.len()) {
            for w in chars.windows(n) {
                let h = fnv1a_chars(w);
                let sign = if h & 1 == 0 { 1.0f32 } else { -1.0 };
                let bin = ((h >> 1) as usize) % d;
                out[bin] += sign;
            }
        }
        l2_normalize(&mut out[..d]);
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = 0.0f32;
    for i in 0..n {
        acc += a[i] * b[i];
    }
    acc.clamp(-1.0, 1.0)
}

fn l2_normalize(v: &mut [f32]) {
    let mut n = 0.0f32;
    for x in v.iter() {
        n += *x * *x;
    }
    let n = n.sqrt();
    if n < 1e-8 {
        return;
    }
    for x in v.iter_mut() {
        *x /= n;
    }
}

fn fnv1a_chars(chars: &[char]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &c in chars {
        let u = c as u32;
        h ^= u as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
        h ^= (u >> 8) as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
        h ^= (u >> 16) as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

#[derive(Debug, Clone)]
pub struct TopicTracker {
    dim: usize,
    k: usize,
    window: Vec<Vec<f32>>,
}

impl TopicTracker {
    pub fn new(dim: usize, k: usize) -> Self {
        Self {
            dim,
            k: k.max(1),
            window: Vec::new(),
        }
    }

    pub fn push(&mut self, v: &[f32]) {
        let mut owned = vec![0.0f32; self.dim];
        let n = self.dim.min(v.len());
        owned[..n].copy_from_slice(&v[..n]);
        self.window.push(owned);
        if self.window.len() > self.k {
            self.window.remove(0);
        }
    }

    pub fn mean(&self, out: &mut [f32]) {
        let d = self.dim.min(out.len());
        out[..d].fill(0.0);
        if self.window.is_empty() {
            return;
        }
        let inv = 1.0 / self.window.len() as f32;
        for v in &self.window {
            for i in 0..d {
                out[i] += v[i] * inv;
            }
        }
        l2_normalize(&mut out[..d]);
    }

    pub fn len(&self) -> usize {
        self.window.len()
    }

    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_cosine_is_one() {
        let e = HashEmbedder::new(64);
        let mut a = vec![0.0; 64];
        e.embed("こんにちは世界", &mut a);
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn related_beats_unrelated() {
        let e = HashEmbedder::new(128);
        let mut t = vec![0.0; 128];
        let mut a = vec![0.0; 128];
        let mut b = vec![0.0; 128];
        e.embed("今日の天気は晴れ", &mut t);
        e.embed("明日の天気も晴れ", &mut a);
        e.embed("xml parser buffer overflow", &mut b);
        assert!(cosine(&t, &a) > cosine(&t, &b));
    }
}
