use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::explain::PathKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Bot,
    /// Preference / control line. Not dialogue, not corpus. Closed analog of RLHF labels.
    Meta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub v: u8,
    pub t: u64,
    pub role: Role,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slipped: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    /// Whether this utterance was absorbed into the generative corpus.
    /// Missing field (old seed logs) means learned — the file is a corpus.
    #[serde(default = "default_learned")]
    pub learned: bool,
    /// Winning source. Written on bot turns so the path mix survives reopen.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<PathKind>,
    /// Token LCS against prior absorbed utterances (bot turns).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub novelty_lcs: Option<usize>,
    /// Chosen candidate length in tokens (for rote reconstruction).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_tok: Option<usize>,
    /// ChaCha position after this reply, split into low/high words so the
    /// full 68-bit stream offset round-trips through ordinary JSON numbers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rng_word_pos: Option<[u64; 2]>,
}

fn default_learned() -> bool {
    true
}

impl Record {
    pub fn user(text: String, learned: bool) -> Self {
        Self {
            v: 1,
            t: now_ms(),
            role: Role::User,
            text,
            slipped: None,
            score: None,
            learned,
            path: None,
            novelty_lcs: None,
            n_tok: None,
            rng_word_pos: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bot(
        text: String,
        learned: bool,
        score: f32,
        slipped: bool,
        path: PathKind,
        novelty_lcs: usize,
        n_tok: usize,
    ) -> Self {
        Self {
            v: 1,
            t: now_ms(),
            role: Role::Bot,
            text,
            slipped: Some(slipped),
            score: Some(score),
            learned,
            path: Some(path),
            novelty_lcs: Some(novelty_lcs),
            n_tok: Some(n_tok),
            rng_word_pos: None,
        }
    }

    /// Preference line (`good` / `bad`). Never enters the corpus.
    pub fn meta(text: String, path: Option<PathKind>) -> Self {
        Self {
            v: 1,
            t: now_ms(),
            role: Role::Meta,
            text,
            slipped: None,
            score: None,
            learned: false,
            path,
            novelty_lcs: None,
            n_tok: None,
            rng_word_pos: None,
        }
    }

    pub(crate) fn set_rng_word_pos(&mut self, pos: u128) {
        self.rng_word_pos = Some([pos as u64, (pos >> 64) as u64]);
    }

    pub(crate) fn saved_rng_word_pos(&self) -> Option<u128> {
        self.rng_word_pos
            .map(|[lo, hi]| u128::from(lo) | (u128::from(hi) << 64))
    }
}

/// Append-only JSONL conversation log. The file is the source of truth;
/// the in-memory index is rebuilt from it.
///
/// `records` stays resident only without a file (ephemeral / from_text):
/// file-backed engines hand the parsed records to `Engine::open` once via
/// `take_records` and never put them back — 2.7M records ≈ 450MB resident
/// otherwise, alone over the 256MB budget.
pub struct AppendLog {
    path: Option<PathBuf>,
    file: Option<File>,
    pub records: Vec<Record>,
}

impl AppendLog {
    /// Open for append without parsing existing records (snapshot loads skip
    /// the parse; the snapshot carries everything the records would feed).
    pub(crate) fn open_no_parse(path: &Path) -> Result<Self> {
        let mut log = Self::open_inner(path, false)?;
        log.records = Vec::new();
        Ok(log)
    }

    pub fn open(path: Option<&Path>) -> Result<Self> {
        let Some(path) = path else {
            return Ok(Self {
                path: None,
                file: None,
                records: Vec::new(),
            });
        };
        Self::open_inner(path, true)
    }

    fn open_inner(path: &Path, parse: bool) -> Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
            }
        }
        let mut records = Vec::new();
        if parse && path.exists() {
            let f = File::open(path).map_err(|e| Error::io(path, e))?;
            for line in BufReader::new(f).lines() {
                let line = line.map_err(|e| Error::io(path, e))?;
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<Record>(&line) {
                    Ok(r) => records.push(r),
                    Err(_) => continue,
                }
            }
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| Error::io(path, e))?;
        // Crash-window repair: a torn final line (no trailing newline) is
        // skipped by the parser above, but appending directly after it would
        // glue the next record onto the torn bytes and silently lose it.
        // Terminate the torn line first.
        if let Ok(meta) = file.metadata() {
            if meta.len() > 0 {
                use std::io::{Read, Seek, SeekFrom};
                let mut f = File::open(path).map_err(|e| Error::io(path, e))?;
                f.seek(SeekFrom::End(-1)).map_err(|e| Error::io(path, e))?;
                let mut last = [0u8; 1];
                f.read_exact(&mut last).map_err(|e| Error::io(path, e))?;
                if last[0] != b'\n' {
                    file.write_all(b"\n").map_err(|e| Error::io(path, e))?;
                    file.sync_data().map_err(|e| Error::io(path, e))?;
                }
            }
        }
        Ok(Self {
            path: Some(path.to_path_buf()),
            file: Some(file),
            records,
        })
    }

    /// In-memory log from JSONL text (browser/embedding entry: no
    /// filesystem). Appends stay in memory; `export_text` round-trips.
    pub fn from_text(text: &str) -> Self {
        let records = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str::<Record>(l).ok())
            .collect();
        Self {
            path: None,
            file: None,
            records,
        }
    }

    /// Serialize all records as JSONL (persistence for hosts without files).
    /// File-backed logs read the file — records are not resident there.
    pub fn export_text(&self) -> String {
        if let Some(p) = &self.path {
            return std::fs::read_to_string(p).unwrap_or_default();
        }
        let mut out = String::new();
        for r in &self.records {
            if let Ok(line) = serde_json::to_string(r) {
                out.push_str(&line);
                out.push('\n');
            }
        }
        out
    }

    pub fn file_backed(&self) -> bool {
        self.file.is_some()
    }

    /// Move the parsed records out (open-time scans borrow them once). The
    /// in-memory caller puts them back; the file-backed caller drops them.
    pub fn take_records(&mut self) -> Vec<Record> {
        std::mem::take(&mut self.records)
    }

    /// Re-read every record from the file (retokenize on file-backed logs).
    pub fn read_all(&self) -> Result<Vec<Record>> {
        let Some(path) = &self.path else {
            return Ok(self.records.clone());
        };
        let f = File::open(path).map_err(|e| Error::io(path, e))?;
        let mut out = Vec::new();
        for line in BufReader::new(f).lines() {
            let line = line.map_err(|e| Error::io(path, e))?;
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(r) = serde_json::from_str::<Record>(&line) {
                out.push(r);
            }
        }
        Ok(out)
    }

    /// Append a user/bot turn as two records with one write and one fsync.
    /// File contents are identical to two `append` calls and the respond path
    /// pays half the sync cost. JSONL does not make the two records atomic
    /// against a torn write; recovery only accepts complete lines.
    pub fn append_turn(&mut self, a: Record, b: Record) -> Result<()> {
        if let Some(f) = self.file.as_mut() {
            let mut lines = serde_json::to_string(&a)?;
            lines.push('\n');
            lines.push_str(&serde_json::to_string(&b)?);
            lines.push('\n');
            f.write_all(lines.as_bytes())
                .map_err(|e| Error::io(self.path.clone().unwrap_or_default(), e))?;
            f.sync_data()
                .map_err(|e| Error::io(self.path.clone().unwrap_or_default(), e))?;
        }
        if self.file.is_none() {
            self.records.push(a);
            self.records.push(b);
        }
        Ok(())
    }

    pub fn append(&mut self, rec: Record) -> Result<()> {
        if let Some(f) = self.file.as_mut() {
            let mut line = serde_json::to_string(&rec)?;
            line.push('\n');
            f.write_all(line.as_bytes())
                .map_err(|e| Error::io(self.path.clone().unwrap_or_default(), e))?;
            f.sync_data()
                .map_err(|e| Error::io(self.path.clone().unwrap_or_default(), e))?;
        }
        if self.file.is_none() {
            self.records.push(rec);
        }
        Ok(())
    }
}

/// Host-injected clock, milliseconds. 0 = unset (use the OS clock). wasm32
/// has no `SystemTime`, so embeddings there must call `set_now_ms` per turn.
static CLOCK_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn set_now_ms(ms: u64) {
    CLOCK_MS.store(ms, std::sync::atomic::Ordering::Relaxed);
}

pub fn now_ms() -> u64 {
    let o = CLOCK_MS.load(std::sync::atomic::Ordering::Relaxed);
    if o != 0 {
        return o;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
    #[cfg(target_arch = "wasm32")]
    {
        0
    }
}
