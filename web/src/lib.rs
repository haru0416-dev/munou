//! Raw-ABI wasm shim: one engine per module instance, strings passed through
//! two thread-local buffers (IN for host→engine, OUT for engine→host). No
//! wasm-bindgen — the JS glue is ~30 lines and the root dependency tree
//! stays untouched. wasm is single-threaded, so thread_local is the whole
//! synchronization story.
//!
//! Call order per turn: `set_now(Date.now())` → write input via `in_ptr` →
//! `respond()` → read OUT. The clock injection is what keeps 日和 / 節目
//! working in the browser (wasm32 has no SystemTime).

use std::cell::RefCell;

use munou_engine::{set_now_ms, Engine, Params};

thread_local! {
    static ENGINE: RefCell<Option<Engine>> = const { RefCell::new(None) };
    static IN: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static OUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn take_in() -> String {
    IN.with(|b| String::from_utf8_lossy(&b.borrow()).into_owned())
}

fn set_out(s: String) {
    OUT.with(|b| *b.borrow_mut() = s.into_bytes());
}

fn with_engine<F: FnOnce(&mut Engine) -> Result<String, String>>(f: F) -> i32 {
    ENGINE.with(|e| match e.borrow_mut().as_mut() {
        None => {
            set_out("engine not initialised".into());
            1
        }
        Some(eng) => match f(eng) {
            Ok(s) => {
                set_out(s);
                0
            }
            Err(s) => {
                set_out(s);
                1
            }
        },
    })
}

/// Host writes `len` input bytes at the returned pointer, then calls the
/// consuming export. The buffer is reused; only valid until the next call.
#[no_mangle]
pub extern "C" fn in_ptr(len: usize) -> *mut u8 {
    IN.with(|b| {
        let mut b = b.borrow_mut();
        b.clear();
        b.resize(len, 0);
        b.as_mut_ptr()
    })
}

#[no_mangle]
pub extern "C" fn out_ptr() -> *const u8 {
    OUT.with(|b| b.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn out_len() -> usize {
    OUT.with(|b| b.borrow().len())
}

/// Milliseconds since epoch, from JS `Date.now()`. Call before init and
/// before every respond — wasm32 has no clock of its own.
#[no_mangle]
pub extern "C" fn set_now(ms: f64) {
    set_now_ms(ms as u64);
}

/// Build the engine from the JSONL log in IN (may be empty). f64 seed: JS
/// numbers are exact to 2^53, wider than any seed the page offers.
#[no_mangle]
pub extern "C" fn init(seed: f64) -> i32 {
    let text = take_in();
    match Engine::open_from_text(Params::default(), seed as u64, &text) {
        Ok(engine) => {
            ENGINE.with(|e| *e.borrow_mut() = Some(engine));
            set_out(String::new());
            0
        }
        Err(err) => {
            set_out(err.to_string());
            1
        }
    }
}

/// Trigger dictionary from the JSON array in IN.
#[no_mangle]
pub extern "C" fn load_triggers() -> i32 {
    let text = take_in();
    with_engine(|e| {
        e.load_triggers_json(&text)
            .map(|_| String::new())
            .map_err(|err| err.to_string())
    })
}

/// One turn: input text in IN, reply JSON in OUT
/// ({interject, text, milestone, strip, why}).
#[no_mangle]
pub extern "C" fn respond() -> i32 {
    let input = take_in();
    with_engine(|e| {
        let r = e.respond(&input).map_err(|err| err.to_string())?;
        let strip = e.observe().strip();
        Ok(serde_json::json!({
            "interject": r.interject,
            "text": r.text,
            "milestone": r.milestone,
            "strip": strip,
            "why": r.trace.explain_text(),
        })
        .to_string())
    })
}

#[no_mangle]
pub extern "C" fn observe() -> i32 {
    with_engine(|e| Ok(e.observe().panel()))
}

#[no_mangle]
pub extern "C" fn ayumi() -> i32 {
    with_engine(|e| Ok(e.ayumi_text()))
}

#[no_mangle]
pub extern "C" fn why() -> i32 {
    with_engine(|e| Ok(e.why_text()))
}

/// `/good` (1) or `/bad` (0) on the last reply.
#[no_mangle]
pub extern "C" fn feedback(good: i32) -> i32 {
    with_engine(|e| e.feedback(good != 0).map_err(|err| err.to_string()))
}

/// Whole log as JSONL, for localStorage persistence. Reopening from this
/// text replays to the identical state (the reproducibility contract).
#[no_mangle]
pub extern "C" fn export_log() -> i32 {
    with_engine(|e| Ok(e.export_log()))
}
