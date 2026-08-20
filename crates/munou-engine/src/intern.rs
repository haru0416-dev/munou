use std::sync::Arc;

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::ids::{TokenId, BOS, EOS, FIRST_USER, SEP};

/// String intern pool. User strings map onto dense `TokenId`s starting at [`FIRST_USER`].
/// Each unique string is stored once (`Arc<str>` shared by map and table);
/// the old `Box<str>` version kept two full copies.
#[derive(Debug, Clone)]
pub struct Interner {
    to_id: FxHashMap<Arc<str>, TokenId>,
    to_str: Vec<Arc<str>>,
}

impl Default for Interner {
    fn default() -> Self {
        Self::new()
    }
}

impl Interner {
    pub fn new() -> Self {
        let mut intern = Self {
            to_id: FxHashMap::default(),
            to_str: Vec::new(),
        };
        intern.to_str.resize(FIRST_USER as usize, Arc::from(""));
        intern.to_str[EOS as usize] = Arc::from("<eos>");
        intern.to_str[BOS as usize] = Arc::from("<bos>");
        intern.to_str[SEP as usize] = Arc::from("<sep>");
        intern
    }

    pub fn intern(&mut self, s: &str) -> TokenId {
        if let Some(&id) = self.to_id.get(s) {
            return id;
        }
        let id = (self.to_str.len()) as TokenId;
        let shared: Arc<str> = Arc::from(s);
        self.to_id.insert(shared.clone(), id);
        self.to_str.push(shared);
        id
    }

    pub fn get(&self, id: TokenId) -> &str {
        self.to_str
            .get(id as usize)
            .map(|s| &s[..])
            .unwrap_or("<unk>")
    }

    pub fn resolve(&self, s: &str) -> Option<TokenId> {
        self.to_id.get(s).copied()
    }

    pub fn len(&self) -> usize {
        self.to_str.len()
    }

    pub fn is_empty(&self) -> bool {
        self.to_str.len() <= FIRST_USER as usize
    }

    pub fn vocab_user(&self) -> usize {
        self.to_str.len().saturating_sub(FIRST_USER as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_roundtrip() {
        let mut a = Interner::new();
        let id = a.intern("天気");
        let mut b = Interner::from_snapshot(a.snapshot());
        assert_eq!(b.get(id), "天気");
        assert_eq!(b.intern("天気"), id);
    }
}

/// Snapshot used when serialising an on-disk index.
#[derive(Debug, Serialize, Deserialize)]
pub struct InternerSnap {
    pub strings: Vec<String>,
}

impl Interner {
    pub fn snapshot(&self) -> InternerSnap {
        InternerSnap {
            strings: self.to_str.iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn from_snapshot(snap: InternerSnap) -> Self {
        let mut intern = Self {
            to_id: FxHashMap::default(),
            to_str: Vec::with_capacity(snap.strings.len().max(FIRST_USER as usize)),
        };
        for (i, s) in snap.strings.into_iter().enumerate() {
            let shared: Arc<str> = Arc::from(s.as_str());
            if i >= FIRST_USER as usize && !shared.is_empty() {
                intern.to_id.insert(shared.clone(), i as TokenId);
            }
            intern.to_str.push(shared);
        }
        while intern.to_str.len() < FIRST_USER as usize {
            intern.to_str.push(Arc::from(""));
        }
        intern.to_str[EOS as usize] = Arc::from("<eos>");
        intern.to_str[BOS as usize] = Arc::from("<bos>");
        intern.to_str[SEP as usize] = Arc::from("<sep>");
        intern
    }
}
