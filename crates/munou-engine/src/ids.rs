//! Dense interned token identifiers.
//!
//! Specials occupy the low range so the suffix-array alphabet stays compact.

pub type TokenId = u32;

pub const EOS: TokenId = 0;
pub const BOS: TokenId = 1;
pub const SEP: TokenId = 2;
pub const FIRST_USER: TokenId = 16;

pub fn is_special(id: TokenId) -> bool {
    id < FIRST_USER
}

pub fn special_name(id: TokenId) -> Option<&'static str> {
    match id {
        EOS => Some("<eos>"),
        BOS => Some("<bos>"),
        SEP => Some("<sep>"),
        _ => None,
    }
}
