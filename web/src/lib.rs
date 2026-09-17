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

use munou_engine::{set_now_ms, Engine, Observe, Params, Stage};

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
            set_out("まだ準備ができていません。".into());
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

fn observation_json(o: Observe) -> serde_json::Value {
    let stage = match o.stage {
        Stage::Empty => "empty",
        Stage::Logged => "logged",
        Stage::Sprout => "sprout",
        Stage::Growing => "growing",
        Stage::Dense => "dense",
    };
    serde_json::json!({
        "stage": stage,
        "utterances": o.utterances,
        "learned": o.learned,
        "tokens": o.tokens,
        "vocab": o.vocab,
        "absorb_rate": o.absorb_rate,
        "eval_n": o.eval_n,
        "band_hit_pct": o.band_hit_pct,
        "rote_lcs": o.rote_lcs,
        "slip_pct": o.slip_pct,
        "band_lo": o.band_lo,
        "band_hi": o.band_hi,
        "last_path": o.last_path,
        "last_learned": o.last_learned,
        "last_slipped": o.last_slipped,
        "last_sim": o.last_sim,
        "recent_learned_bot": o.recent_learned_bot,
        "paths": {
            "trigger": o.path_trig,
            "markov": o.path_mark,
            "retrieve": o.path_retr,
            "echo": o.path_echo,
            "adapt": o.path_adpt,
        },
        "working": o.working,
        "hist": o.hist,
        "meta": o.meta,
        "path_prior": o.path_prior,
        "rote_lean": o.rote_lean,
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
/// ({interject, text, milestone, observation}).
#[no_mangle]
pub extern "C" fn respond() -> i32 {
    let input = take_in();
    with_engine(|e| {
        let r = e.respond(&input).map_err(|err| err.to_string())?;
        let observation = observation_json(e.observe());
        let mut reply = serde_json::json!({
            "interject": r.interject,
            "text": r.text,
            "milestone": r.milestone,
        });
        reply["observation"] = observation;
        Ok(reply.to_string())
    })
}

#[no_mangle]
pub extern "C" fn observe() -> i32 {
    with_engine(|e| Ok(observation_json(e.observe()).to_string()))
}

#[no_mangle]
pub extern "C" fn history() -> i32 {
    with_engine(|e| {
        let history = serde_json::to_string(&e.history()).map_err(|err| err.to_string())?;
        let observation = observation_json(e.observe());
        Ok(format!(
            "{{\"history\":{history},\"observation\":{observation}}}"
        ))
    })
}

#[no_mangle]
pub extern "C" fn why() -> i32 {
    with_engine(|e| {
        let trace = serde_json::to_string(&e.last_trace()).map_err(|err| err.to_string())?;
        let observation = observation_json(e.observe());
        Ok(format!(
            "{{\"trace\":{trace},\"observation\":{observation}}}"
        ))
    })
}

/// `/good` (1) or `/bad` (0) on the last reply.
#[no_mangle]
pub extern "C" fn feedback(good: i32) -> i32 {
    with_engine(|e| {
        if e.last_trace().is_none() && e.observe().last_path.is_none() {
            return Err("まだ評価できる返答がありません。".into());
        }
        e.feedback(good != 0).map_err(|err| err.to_string())?;
        Ok(if good != 0 {
            "この返し方を少し選びやすくしました。"
        } else {
            "この返し方を少し控えるようにしました。"
        }
        .into())
    })
}

/// Whole log as JSONL, for localStorage persistence. Reopening from this
/// text replays to the identical state (the reproducibility contract).
#[no_mangle]
pub extern "C" fn export_log() -> i32 {
    with_engine(|e| Ok(e.export_log()))
}
