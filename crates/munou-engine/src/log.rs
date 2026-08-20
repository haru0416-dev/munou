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
    /// Winning source. Written on bot turns from v0.1.3 so path mix survives reopen.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<PathKind>,
    /// Token LCS against prior absorbed utterances (bot turns).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub novelty_lcs: Option<usize>,
    /// Chosen candidate length in tokens (for rote reconstruction).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_tok: Option<usize>,
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
        }
    }
}

/// Append-only JSONL conversation log. The file is the source of truth;
/// the in-memory index is rebuilt from it.
pub struct AppendLog {
    path: Option<PathBuf>,
    file: Option<File>,
    pub records: Vec<Record>,
}

impl AppendLog {
    pub fn open(path: Option<&Path>) -> Result<Self> {
        let Some(path) = path else {
            return Ok(Self {
                path: None,
                file: None,
                records: Vec::new(),
            });
        };
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
            }
        }
        let mut records = Vec::new();
        if path.exists() {
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
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| Error::io(path, e))?;
        Ok(Self {
            path: Some(path.to_path_buf()),
            file: Some(file),
            records,
        })
    }

    /// Append a user/bot turn as two records with one write and one fsync.
    /// File contents are identical to two `append` calls; the turn becomes
    /// atomic on disk and the respond path pays half the sync cost.
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
        self.records.push(a);
        self.records.push(b);
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
        self.records.push(rec);
        Ok(())
    }
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
