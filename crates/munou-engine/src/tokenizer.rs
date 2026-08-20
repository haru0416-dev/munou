//! Unsupervised Japanese-oriented tokenizer.
//!
//! Two layers:
//! 1. **Morphemes** — Unicode script runs, then **bidirectional** branching-entropy
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
        // prolonged sound mark: base class Kata; `effective_classes` lets it
        // inherit a CJK neighbour so らーめん is one run, not ら|ー|めん
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

/// Successor histogram. Up to `INLINE` distinct successors live inline;
/// only wider contexts pay for a boxed map. The per-context fixed cost is
/// what dominated resident memory (~200B with boxed keys + per-context map;
/// ~56B packed), 246MB at 1.5M tokens before this layout.
#[derive(Default, Debug, Clone)]
struct NextStats {
    n_inline: u8,
    inline: [(u32, u32); INLINE],
    /// Sorted by char — a canonical order, so the entropy float sum is
    /// identical however the histogram was built (replay or snapshot).
    /// Boxed to keep the no-spill NextStats at one pointer, not three words.
    #[allow(clippy::box_collection)]
    spill: Option<Box<Vec<(u32, u32)>>>,
}

const INLINE: usize = 3;

impl NextStats {
    fn bump(&mut self, ch: u32, k: u32) {
        if let Some(v) = self.spill.as_mut() {
            match v.binary_search_by_key(&ch, |e| e.0) {
                Ok(i) => v[i].1 += k,
                Err(i) => v.insert(i, (ch, k)),
            }
            return;
        }
        for e in self.inline[..self.n_inline as usize].iter_mut() {
            if e.0 == ch {
                e.1 += k;
                return;
            }
        }
        if (self.n_inline as usize) < INLINE {
            self.inline[self.n_inline as usize] = (ch, k);
            self.n_inline += 1;
            return;
        }
        let mut v: Vec<(u32, u32)> = self.inline.to_vec();
        v.push((ch, k));
        v.sort_unstable_by_key(|e| e.0);
        self.spill = Some(Box::new(v));
    }

    /// Shannon entropy of the successor histogram. Counts equal the boxed-key
    /// representation exactly; the float summation order is canonical (inline
    /// insertion order, spill sorted), so cut decisions within one ulp of the
    /// threshold may differ from the pre-v0.1.16 layout (declared as a
    /// reply-sequence change, transcript diff 0 in practice).
    fn entropy(&self) -> f64 {
        match self.spill.as_ref() {
            Some(v) => entropy_of(v.iter().map(|&(_, c)| c)),
            None => entropy_of(
                self.inline[..self.n_inline as usize]
                    .iter()
                    .map(|&(_, c)| c),
            ),
        }
    }

    /// Successors in container order (canonical: inline insertion order,
    /// spill sorted) — reloading in this order reproduces the container and
    /// therefore the entropy float sums exactly.
    fn successors(&self) -> Vec<(u32, u32)> {
        match self.spill.as_ref() {
            Some(v) => v.as_ref().clone(),
            None => self.inline[..self.n_inline as usize].to_vec(),
        }
    }

    fn from_successors(list: Vec<(u32, u32)>) -> Self {
        let mut st = NextStats::default();
        if list.len() <= INLINE {
            for (i, e) in list.into_iter().enumerate() {
                st.inline[i] = e;
                st.n_inline = i as u8 + 1;
            }
        } else {
            st.spill = Some(Box::new(list));
        }
        st
    }
}

fn entropy_of<I: Iterator<Item = u32> + Clone>(counts: I) -> f64 {
    let total: u32 = counts.clone().sum();
    if total == 0 {
        return 0.0;
    }
    let t = total as f64;
    let mut h = 0.0;
    for c in counts {
        if c == 0 {
            continue;
        }
        let p = c as f64 / t;
        h -= p * p.ln();
    }
    h
}

/// N-gram key: (char+1) packed into 21-bit fields, zero-padded — unambiguous
/// for n ≤ 6 (Unicode scalars fit 21 bits). One u128 replaces a heap
/// `Box<[u32]>` per context.
fn pack(gram: &[u32]) -> u128 {
    let mut k = 0u128;
    for &c in gram {
        k = (k << 21) | (c as u128 + 1);
    }
    k
}

/// (packed key, forward successors, backward predecessors) per entry.
pub(crate) type EntropyEntries = Vec<(u128, Vec<(u32, u32)>, Vec<(u32, u32)>)>;

/// See `EntropyModel::snap_dump`.
pub(crate) struct EntropySnap {
    pub max_n: usize,
    pub total_chars: u64,
    pub entries: EntropyEntries,
}

/// Both directions of one context. Sharing the map entry halves the key +
/// control overhead — nearly every observed gram carries both sides.
#[derive(Debug, Clone, Default)]
struct Ctx {
    fwd: NextStats,
    bwd: NextStats,
}

/// Character n-gram statistics collected from the conversation log only.
#[derive(Debug, Clone, Default)]
pub struct EntropyModel {
    map: FxHashMap<u128, Ctx>,
    max_n: usize,
    total_chars: u64,
}

impl EntropyModel {
    pub fn new(max_n: usize) -> Self {
        Self {
            map: FxHashMap::default(),
            // Upper clamp is the packing width; the parameter has no CLI
            // surface and the default is 5.
            max_n: max_n.clamp(2, 6),
            total_chars: 0,
        }
    }

    pub fn observe(&mut self, text: &str) {
        self.observe_n(text, 1);
    }

    /// Observe `text` as if it appeared `k` times. Additive counts make this
    /// bit-identical to `k` plain observes — the closed log repeats the same
    /// lines heavily (fabricate unique0: 46 distinct in 200k), so replay
    /// groups by text and pays one hash pass per distinct line.
    pub fn observe_n(&mut self, text: &str, k: u32) {
        if k == 0 {
            return;
        }
        let chars: Vec<u32> = text.chars().map(|c| c as u32).collect();
        self.total_chars += chars.len() as u64 * k as u64;
        for n in 1..=self.max_n {
            if chars.len() < n {
                continue;
            }
            for i in 0..=chars.len() - n {
                let e = self.map.entry(pack(&chars[i..i + n])).or_default();
                if i > 0 {
                    e.bwd.bump(chars[i - 1], k);
                }
                if i + n < chars.len() {
                    e.fwd.bump(chars[i + n], k);
                }
            }
        }
    }

    pub fn ready(&self) -> bool {
        self.total_chars >= 256
    }

    /// (contexts, spilled histograms, estimated heap bytes).
    pub(crate) fn mem_stats(&self) -> (usize, usize, usize) {
        let mut spill = 0usize;
        let mut heap = self.map.capacity() * (std::mem::size_of::<(u128, Ctx)>() + 1);
        for c in self.map.values() {
            for st in [&c.fwd, &c.bwd] {
                if let Some(v) = &st.spill {
                    spill += 1;
                    heap += 24 + v.capacity() * 8;
                }
            }
        }
        (self.map.len(), spill, heap)
    }

    /// Snapshot payload. Map order is irrelevant (lookup by key only);
    /// per-context successor order is canonical and round-trips exactly.
    pub(crate) fn snap_dump(&self) -> EntropySnap {
        EntropySnap {
            max_n: self.max_n,
            total_chars: self.total_chars,
            entries: self
                .map
                .iter()
                .map(|(k, c)| (*k, c.fwd.successors(), c.bwd.successors()))
                .collect(),
        }
    }

    /// Streaming load counterpart of `snap_dump`: entries insert one at a
    /// time so the loader never holds a second full copy of the model.
    pub(crate) fn snap_new(max_n: usize, total_chars: u64, capacity: usize) -> Self {
        let mut map = FxHashMap::default();
        map.reserve(capacity);
        Self {
            map,
            max_n,
            total_chars,
        }
    }

    pub(crate) fn snap_insert(&mut self, k: u128, fwd: Vec<(u32, u32)>, bwd: Vec<(u32, u32)>) {
        self.map.insert(
            k,
            Ctx {
                fwd: NextStats::from_successors(fwd),
                bwd: NextStats::from_successors(bwd),
            },
        );
    }

    fn right_entropy(&self, gram: &[u32]) -> f64 {
        self.map
            .get(&pack(gram))
            .map(|c| c.fwd.entropy())
            .unwrap_or(0.0)
    }

    fn left_entropy(&self, gram: &[u32]) -> f64 {
        self.map
            .get(&pack(gram))
            .map(|c| c.bwd.entropy())
            .unwrap_or(0.0)
    }

    fn cut_score(&self, chars: &[u32], i: usize) -> f64 {
        // boundary *after* index i (between i and i+1)
        let n = self.max_n;
        let left_n = n.min(i + 1);
        let hr = if left_n == 0 {
            0.0
        } else {
            self.right_entropy(&chars[i + 1 - left_n..=i])
        };
        let rest = chars.len().saturating_sub(i + 1);
        let right_n = n.min(rest);
        let hl = if right_n == 0 {
            0.0
        } else {
            self.left_entropy(&chars[i + 1..i + 1 + right_n])
        };
        0.5 * hr + 0.5 * hl
    }
}

#[derive(Debug, Clone)]
pub struct Tokenized {
    pub morphemes: Vec<TokenId>,
    pub chunks: Vec<TokenId>,
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

    pub(crate) fn model(&self) -> &EntropyModel {
        &self.model
    }

    pub(crate) fn with_model(params: &Params, model: EntropyModel) -> Self {
        Self {
            model,
            cut: params.entropy_cut,
            chunk_morphs: params.chunk_morphs.max(1),
        }
    }

    /// Weighted observe for replay: identical counts to `k` plain observes.
    pub(crate) fn observe_n(&mut self, text: &str, k: u32) {
        self.model.observe_n(text, k);
    }

    pub fn tokenize(&self, intern: &mut Interner, text: &str) -> Tokenized {
        let spans = self.morph_spans(text);
        let mut morphemes = Vec::with_capacity(spans.len());
        for &(lo, hi) in &spans {
            morphemes.push(intern.intern(&text[lo..hi]));
        }
        let chunks = chunk_intern(intern, &morphemes, self.chunk_morphs);
        Tokenized { morphemes, chunks }
    }

    fn morph_spans(&self, text: &str) -> Vec<(usize, usize)> {
        let char_idx: Vec<(usize, char)> = text.char_indices().collect();
        if char_idx.is_empty() {
            return Vec::new();
        }
        let n = char_idx.len();
        let cls = effective_classes(&char_idx);
        let mut cuts = vec![false; n + 1];
        cuts[0] = true;
        cuts[n] = true;
        for i in 1..n {
            if cls[i - 1] != cls[i] {
                cuts[i] = true;
            }
        }
        if self.model.ready() {
            let codes: Vec<u32> = char_idx.iter().map(|(_, c)| *c as u32).collect();
            let mut scores = vec![0.0; n];
            for i in 0..n.saturating_sub(1) {
                if cls[i] == cls[i + 1] && is_cjk(cls[i]) {
                    scores[i] = self.model.cut_score(&codes, i);
                }
            }
            for i in 0..n.saturating_sub(1) {
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
            for (i, &c) in cls.iter().enumerate() {
                if is_cjk(c) {
                    cuts[i] = true;
                    cuts[i + 1] = true;
                }
            }
        }
        let byte_at = |i: usize| -> usize {
            if i < n {
                char_idx[i].0
            } else {
                text.len()
            }
        };
        let mut out = Vec::new();
        let mut start = 0;
        #[allow(clippy::needless_range_loop)]
        for i in 1..=n {
            if cuts[i] {
                if i > start {
                    let lo = byte_at(start);
                    let hi = byte_at(i);
                    let s = &text[lo..hi];
                    if !s.chars().all(|c| class(c) == Class::Ws) {
                        out.push((lo, hi));
                    }
                }
                start = i;
            }
        }
        out
    }

    #[cfg(test)]
    fn morphemes(&self, text: &str) -> Vec<String> {
        self.morph_spans(text)
            .into_iter()
            .map(|(lo, hi)| text[lo..hi].to_string())
            .collect()
    }
}

fn is_cjk(c: Class) -> bool {
    matches!(c, Class::Han | Class::Hira | Class::Kata)
}

/// True when every char is punctuation. Punctuation chunks are standalone
/// tokens, so generation can start a reply with 「、」; the engine trims those.
pub(crate) fn is_punct_str(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| class(c) == Class::Punct)
}

/// Per-char classes with the prolonged sound mark inheriting a CJK
/// neighbour's class (preferring the left one). The plain `class` pinned ー
/// to katakana, which forced boundary cuts like ら|ー|めん forever, no
/// matter what the entropy model had learned.
fn effective_classes(char_idx: &[(usize, char)]) -> Vec<Class> {
    let mut cls: Vec<Class> = char_idx.iter().map(|&(_, c)| class(c)).collect();
    for i in 0..cls.len() {
        if char_idx[i].1 != 'ー' {
            continue;
        }
        let prev = if i > 0 { Some(cls[i - 1]) } else { None };
        let next = char_idx.get(i + 1).map(|&(_, c)| class(c));
        cls[i] = match (prev, next) {
            (Some(p), _) if is_cjk(p) => p,
            (_, Some(nx)) if is_cjk(nx) => nx,
            _ => Class::Kata,
        };
    }
    cls
}

fn chunk_intern(intern: &mut Interner, morphs: &[TokenId], k: usize) -> Vec<TokenId> {
    if morphs.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut n = 0;
    for &id in morphs {
        let punct = intern.get(id).chars().all(|c| class(c) == Class::Punct);
        if punct {
            if !buf.is_empty() {
                out.push(intern.intern(&buf));
                buf.clear();
                n = 0;
            }
            out.push(id);
            continue;
        }
        buf.push_str(intern.get(id));
        n += 1;
        if n >= k {
            out.push(intern.intern(&buf));
            buf.clear();
            n = 0;
        }
    }
    if !buf.is_empty() {
        out.push(intern.intern(&buf));
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
        let morphs: Vec<&str> = t.morphemes.iter().map(|id| intern.get(*id)).collect();
        assert!(morphs.contains(&"hello"));
        assert!(morphs.iter().any(|s| s.contains('世') || *s == "世"));
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

    /// ー between hiragana must not force class-boundary cuts; with a trained
    /// model and zero in-word branching entropy, らーめん stays one morpheme
    /// (a fixed katakana class for ー splits ら|ー|めん unconditionally).
    #[test]
    fn prolonged_mark_joins_neighbouring_run() {
        let mut tok = Tokenizer::new(&Params::default());
        for _ in 0..80 {
            tok.observe("らーめんたべたい。らーめんおいしい。");
        }
        let morphs = tok.morphemes("らーめん");
        assert!(
            morphs.iter().all(|m| m != "ー"),
            "ー must not be a lone morpheme: {morphs:?}"
        );
        assert!(
            morphs.iter().any(|m| m.contains("らー")),
            "run should cross the prolonged mark: {morphs:?}"
        );
    }

    /// Weighted observe must reproduce k repeated observes exactly — the
    /// segmentation (which depends only on the counts) must not differ.
    #[test]
    fn observe_n_equals_repeated_observe() {
        let corpus = "東京都の天気は晴れ。大阪府の天気は雨。らーめんたべたい。";
        let mut a = Tokenizer::new(&Params::default());
        for _ in 0..40 {
            a.observe(corpus);
        }
        let mut b = Tokenizer::new(&Params::default());
        b.observe_n(corpus, 40);
        for probe in ["東京都の天気は晴れ", "らーめん", corpus] {
            assert_eq!(a.morphemes(probe), b.morphemes(probe), "probe={probe}");
        }
    }

    #[test]
    fn detokenize_ascii_spacing() {
        assert_eq!(detokenize(&["hello".into(), "world".into()]), "hello world");
        assert_eq!(detokenize(&["今日".into(), "は".into()]), "今日は");
    }
}
