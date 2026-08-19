//! Unsupervised Japanese-oriented tokenizer.
//!
//! Two layers:
//! 1. **Morphemes** — Unicode script runs, then branching-entropy / accessor-variety
//!    cuts inside CJK runs (no dictionary, no labelled data).
//! 2. **Statistical chunks** — adjacent morphemes grouped into Markov units.
//!    We do not insist on bunsetsu; chunks exist to keep Markov slightly grammatical.

use rustc_hash::FxHashMap;

use crate::ids::TokenId;
use crate::intern::Interner;
use crate::params::Params;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    Ws,
    Punct,
    Latin,
    Digit,
    Hira,
    Kata,
    Han,
    Other,
}

fn class(c: char) -> Class {
    if c.is_whitespace() {
        Class::Ws
    } else if c.is_ascii_digit() {
        Class::Digit
    } else if c.is_ascii_alphanumeric() {
        Class::Latin
    } else if matches!(
        c,
        '。' | '、' | '！' | '？' | '…' | '「' | '」' | '『' | '』' | '（' | '）' | '・' | 'ー'
    ) || c.is_ascii_punctuation()
    {
        // prolonged sound mark is kept with katakana via the class of neighbours
        if c == 'ー' {
            Class::Kata
        } else {
            Class::Punct
        }
    } else {
        let u = c as u32;
        if (0x3040..=0x309F).contains(&u) {
            Class::Hira
        } else if (0x30A0..=0x30FF).contains(&u) || (0xFF66..=0xFF9D).contains(&u) {
            Class::Kata
        } else if (0x4E00..=0x9FFF).contains(&u) || (0x3400..=0x4DBF).contains(&u) {
            Class::Han
        } else {
            Class::Other
        }
    }
}

#[derive(Default, Debug, Clone)]
struct NextStats {
    count: u32,
    next: FxHashMap<u32, u32>,
}

/// Character n-gram statistics collected from the conversation log only.
#[derive(Debug, Clone, Default)]
pub struct EntropyModel {
    fwd: FxHashMap<Box<[u32]>, NextStats>,
    max_n: usize,
    total_chars: u64,
}

impl EntropyModel {
    pub fn new(max_n: usize) -> Self {
        Self {
            fwd: FxHashMap::default(),
            max_n: max_n.max(2),
            total_chars: 0,
        }
    }

    pub fn observe(&mut self, text: &str) {
        let chars: Vec<u32> = text.chars().map(|c| c as u32).collect();
        self.total_chars += chars.len() as u64;
        for n in 1..=self.max_n {
            if chars.len() < n {
                continue;
            }
            for i in 0..=chars.len() - n {
                let key: Box<[u32]> = chars[i..i + n].into();
                let e = self.fwd.entry(key).or_default();
                e.count = e.count.saturating_add(1);
                if i + n < chars.len() {
                    *e.next.entry(chars[i + n]).or_insert(0) += 1;
                }
            }
        }
    }

    pub fn ready(&self) -> bool {
        self.total_chars >= 256
    }

    fn right_entropy(&self, gram: &[u32]) -> f64 {
        let Some(st) = self.fwd.get(gram) else {
            return 0.0;
        };
        entropy(&st.next)
    }

    fn accessor_variety(&self, gram: &[u32]) -> f64 {
        let Some(st) = self.fwd.get(gram) else {
            return 0.0;
        };
        (st.next.len() as f64).max(1.0).ln()
    }

    fn cut_score(&self, chars: &[u32], i: usize) -> f64 {
        // boundary *after* index i (between i and i+1)
        let n = self.max_n.min(i + 1);
        if n == 0 {
            return 0.0;
        }
        let gram = &chars[i + 1 - n..=i];
        0.5 * self.right_entropy(gram) + 0.5 * self.accessor_variety(gram)
    }
}

fn entropy(hist: &FxHashMap<u32, u32>) -> f64 {
    let total: u32 = hist.values().copied().sum();
    if total == 0 {
        return 0.0;
    }
    let t = total as f64;
    let mut h = 0.0;
    for &c in hist.values() {
        if c == 0 {
            continue;
        }
        let p = c as f64 / t;
        h -= p * p.ln();
    }
    h
}

#[derive(Debug, Clone)]
pub struct Tokenized {
    pub morphemes: Vec<TokenId>,
    pub chunks: Vec<TokenId>,
    pub morph_strs: Vec<String>,
    pub chunk_strs: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Tokenizer {
    model: EntropyModel,
    cut: f64,
    chunk_morphs: usize,
}

impl Tokenizer {
    pub fn new(params: &Params) -> Self {
        Self {
            model: EntropyModel::new(params.entropy_n),
            cut: params.entropy_cut,
            chunk_morphs: params.chunk_morphs.max(1),
        }
    }

    pub fn observe(&mut self, text: &str) {
        self.model.observe(text);
    }

    pub fn tokenize(&self, intern: &mut Interner, text: &str) -> Tokenized {
        let morphs = self.morphemes(text);
        let chunks = chunk(&morphs, self.chunk_morphs);
        let mut morphemes = Vec::with_capacity(morphs.len());
        let mut morph_strs = Vec::with_capacity(morphs.len());
        for m in &morphs {
            morph_strs.push(m.clone());
            morphemes.push(intern.intern(m));
        }
        let mut chunk_ids = Vec::with_capacity(chunks.len());
        let mut chunk_strs = Vec::with_capacity(chunks.len());
        for c in &chunks {
            chunk_strs.push(c.clone());
            chunk_ids.push(intern.intern(c));
        }
        Tokenized {
            morphemes,
            chunks: chunk_ids,
            morph_strs,
            chunk_strs,
        }
    }

    fn morphemes(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        if chars.is_empty() {
            return Vec::new();
        }
        let mut cuts = vec![false; chars.len() + 1];
        cuts[0] = true;
        cuts[chars.len()] = true;

        for i in 1..chars.len() {
            if class(chars[i - 1]) != class(chars[i]) {
                cuts[i] = true;
            }
        }

        if self.model.ready() {
            let codes: Vec<u32> = chars.iter().map(|c| *c as u32).collect();
            let mut scores = vec![0.0; chars.len()];
            for i in 0..chars.len().saturating_sub(1) {
                if class(chars[i]) == class(chars[i + 1]) && is_cjk(class(chars[i])) {
                    scores[i] = self.model.cut_score(&codes, i);
                }
            }
            for i in 0..chars.len().saturating_sub(1) {
                if scores[i] < self.cut {
                    continue;
                }
                let left = if i == 0 { 0.0 } else { scores[i - 1] };
                let right = scores.get(i + 1).copied().unwrap_or(0.0);
                if scores[i] >= left && scores[i] >= right {
                    cuts[i + 1] = true;
                }
            }
        } else {
            // Weak model: split CJK to characters so Markov has something to chew.
            for (i, &c) in chars.iter().enumerate() {
                if is_cjk(class(c)) {
                    cuts[i] = true;
                    cuts[i + 1] = true;
                }
            }
        }

        let mut out = Vec::new();
        let mut start = 0;
        for i in 1..=chars.len() {
            if cuts[i] {
                if i > start {
                    let s: String = chars[start..i].iter().collect();
                    if !s.chars().all(|c| class(c) == Class::Ws) {
                        out.push(s);
                    }
                }
                start = i;
            }
        }
        out
    }
}

fn is_cjk(c: Class) -> bool {
    matches!(c, Class::Han | Class::Hira | Class::Kata)
}

fn chunk(morphs: &[String], k: usize) -> Vec<String> {
    if morphs.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut n = 0;
    for m in morphs {
        let punct = m.chars().all(|c| class(c) == Class::Punct);
        if punct {
            if !buf.is_empty() {
                out.push(std::mem::take(&mut buf));
                n = 0;
            }
            out.push(m.clone());
            continue;
        }
        buf.push_str(m);
        n += 1;
        if n >= k {
            out.push(std::mem::take(&mut buf));
            n = 0;
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

/// Join chunk strings for display: spaces only between ASCII-letter tokens.
pub fn detokenize(chunks: &[String]) -> String {
    let mut out = String::new();
    for (i, c) in chunks.iter().enumerate() {
        if i > 0 && needs_space(out.chars().last(), c.chars().next()) {
            out.push(' ');
        }
        out.push_str(c);
    }
    out
}

fn needs_space(left: Option<char>, right: Option<char>) -> bool {
    match (left, right) {
        (Some(l), Some(r)) => l.is_ascii_alphanumeric() && r.is_ascii_alphanumeric(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::Params;

    #[test]
    fn splits_script_runs() {
        let tok = Tokenizer::new(&Params::default());
        let mut intern = Interner::new();
        let t = tok.tokenize(&mut intern, "hello世界123");
        assert!(t.morph_strs.iter().any(|s| s == "hello"));
        assert!(t.morph_strs.iter().any(|s| s.contains('世') || *s == "世"));
    }

    #[test]
    fn entropy_cuts_after_observation() {
        let mut tok = Tokenizer::new(&Params::default());
        let corpus = "東京都の天気は晴れ。大阪府の天気は雨。北海道の天気は雪。";
        for _ in 0..40 {
            tok.observe(corpus);
        }
        let morphs = tok.morphemes(corpus);
        assert!(morphs.len() > 3, "got {morphs:?}");
    }

    #[test]
    fn detokenize_ascii_spacing() {
        assert_eq!(detokenize(&["hello".into(), "world".into()]), "hello world");
        assert_eq!(detokenize(&["今日".into(), "は".into()]), "今日は");
    }
}
