use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Bot,
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
