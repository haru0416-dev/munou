//! Snapshot — a pure cache of the replay-equivalent open state.
//!
//! Contract: loading a snapshot must equal a full JSONL replay bit for bit
//! (the spec test pins reply sequences and gauges). Anything that could
//! diverge is keyed out: the header carries a schema revision, a params
//! hash, and the log file's length + FNV — any mismatch falls back to the
//! full build, which rewrites the file. The JSONL log stays the source of
//! truth; deleting snapshot.bin is always safe.
//!
//! Live state is never written: only `Engine::open` writes, right after a
//! full build, when the state *is* the replay state. Writing a live session
//! would make open() depend on cache presence.
//!
//! Format: hand-rolled little-endian, length-prefixed. No new dependencies.

use std::path::{Path, PathBuf};

use crate::ids::TokenId;

/// Bump on ANY change that alters replay outcomes for the same params
/// (tokenizer, store semantics, replay order, …). Release checklist item.
pub(crate) const SNAP_REV: u32 = 1;
const MAGIC: &[u8; 4] = b"MNSN";
/// Logs smaller than this open in milliseconds anyway; skip the file churn.
pub(crate) const MIN_LOG_BYTES: u64 = 4096;

pub(crate) fn snapshot_path(log_path: &Path) -> PathBuf {
    log_path.with_file_name("snapshot.bin")
}

/// FNV-1a over a file, streamed; returns (len, hash).
pub(crate) fn hash_file(path: &Path) -> std::io::Result<(u64, u64)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut buf = [0u8; 65536];
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut len = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        len += n as u64;
        for &b in &buf[..n] {
            h ^= b as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    }
    Ok((len, h))
}

pub(crate) fn fnv64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

// ---- writer ----

pub(crate) struct W(pub Vec<u8>);

impl W {
    pub fn header(params_fnv: u64, log_len: u64, log_fnv: u64) -> Self {
        let mut w = W(Vec::with_capacity(1 << 20));
        w.0.extend_from_slice(MAGIC);
        w.u32(SNAP_REV);
        w.u64(params_fnv);
        w.u64(log_len);
        w.u64(log_fnv);
        w
    }
    pub fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    pub fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u128(&mut self, v: u128) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn f32(&mut self, v: f32) {
        self.u32(v.to_bits());
    }
    pub fn f64(&mut self, v: f64) {
        self.u64(v.to_bits());
    }
    pub fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.0.extend_from_slice(s.as_bytes());
    }
    pub fn toks(&mut self, ts: &[TokenId]) {
        self.u32(ts.len() as u32);
        for &t in ts {
            self.u32(t);
        }
    }
    pub fn opt_str(&mut self, s: Option<&str>) {
        match s {
            None => self.u8(0),
            Some(s) => {
                self.u8(1);
                self.str(s);
            }
        }
    }
    pub fn opt_u64(&mut self, v: Option<u64>) {
        match v {
            None => self.u8(0),
            Some(v) => {
                self.u8(1);
                self.u64(v);
            }
        }
    }
}

// ---- reader ----

pub(crate) struct R<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> R<'a> {
    /// None unless magic / revision / params / log identity all match.
    pub fn open(bytes: &'a [u8], params_fnv: u64, log_len: u64, log_fnv: u64) -> Option<Self> {
        let mut r = R { b: bytes, i: 0 };
        if r.take(4)? != MAGIC {
            return None;
        }
        if r.u32()? != SNAP_REV
            || r.u64()? != params_fnv
            || r.u64()? != log_len
            || r.u64()? != log_fnv
        {
            return None;
        }
        Some(r)
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.b.get(self.i..self.i + n)?;
        self.i += n;
        Some(s)
    }
    pub fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    pub fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    pub fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    pub fn u128(&mut self) -> Option<u128> {
        Some(u128::from_le_bytes(self.take(16)?.try_into().ok()?))
    }
    pub fn f32(&mut self) -> Option<f32> {
        Some(f32::from_bits(self.u32()?))
    }
    pub fn f64(&mut self) -> Option<f64> {
        Some(f64::from_bits(self.u64()?))
    }
    pub fn str(&mut self) -> Option<String> {
        let n = self.u32()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).ok()
    }
    pub fn toks(&mut self) -> Option<Vec<TokenId>> {
        let n = self.u32()? as usize;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(self.u32()?);
        }
        Some(v)
    }
    pub fn opt_str(&mut self) -> Option<Option<String>> {
        match self.u8()? {
            0 => Some(None),
            _ => Some(Some(self.str()?)),
        }
    }
    pub fn opt_u64(&mut self) -> Option<Option<u64>> {
        match self.u8()? {
            0 => Some(None),
            _ => Some(Some(self.u64()?)),
        }
    }
}

/// Atomic write: tmp + rename, best-effort (a failed cache write must never
/// fail the open).
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) {
    let tmp = path.with_extension("bin.tmp");
    if std::fs::write(&tmp, bytes).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_gates_on_identity() {
        let mut w = W::header(1, 2, 3);
        w.str("x");
        assert!(R::open(&w.0, 1, 2, 3).is_some());
        assert!(R::open(&w.0, 9, 2, 3).is_none(), "params mismatch");
        assert!(R::open(&w.0, 1, 9, 3).is_none(), "log length mismatch");
        assert!(R::open(&w.0, 1, 2, 9).is_none(), "log hash mismatch");
        assert!(R::open(&w.0[..8], 1, 2, 3).is_none(), "truncation");
    }

    #[test]
    fn primitives_round_trip() {
        let mut w = W::header(0, 0, 0);
        w.u8(7);
        w.u128(1 << 100);
        w.f64(0.1);
        w.str("こんにちは");
        w.toks(&[1, 2, 3]);
        w.opt_str(None);
        w.opt_u64(Some(42));
        let mut r = R::open(&w.0, 0, 0, 0).unwrap();
        assert_eq!(r.u8(), Some(7));
        assert_eq!(r.u128(), Some(1 << 100));
        assert_eq!(r.f64(), Some(0.1));
        assert_eq!(r.str().as_deref(), Some("こんにちは"));
        assert_eq!(r.toks(), Some(vec![1, 2, 3]));
        assert_eq!(r.opt_str(), Some(None));
        assert_eq!(r.opt_u64(), Some(Some(42)));
    }
}
