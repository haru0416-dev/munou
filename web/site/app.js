// Loader for the raw-ABI wasm shim (no wasm-bindgen). Strings cross the
// boundary through the module's IN/OUT buffers; views into memory.buffer are
// re-created after every call because memory growth detaches them.
"use strict";

const LOG_KEY = "munou.log";
const SEED_KEY = "munou.seed";
const enc = new TextEncoder();
const dec = new TextDecoder();

let wasm = null;

function put(s) {
  const bytes = enc.encode(s);
  const ptr = wasm.in_ptr(bytes.length);
  new Uint8Array(wasm.memory.buffer, ptr, bytes.length).set(bytes);
}

function out() {
  return dec.decode(new Uint8Array(wasm.memory.buffer, wasm.out_ptr(), wasm.out_len()));
}

const chat = document.getElementById("chat");
const input = document.getElementById("input");
const send = document.getElementById("send");
const seedBox = document.getElementById("seed");

function add(cls, text) {
  const p = document.createElement("p");
  p.className = "msg " + cls;
  p.textContent = text;
  chat.appendChild(p);
  p.scrollIntoView({ block: "end" });
  return p;
}

function addPanel(text) {
  const pre = document.createElement("pre");
  pre.className = "panel";
  pre.textContent = text;
  chat.appendChild(pre);
  pre.scrollIntoView({ block: "end" });
}

function persist() {
  if (wasm.export_log() === 0) {
    try { localStorage.setItem(LOG_KEY, out()); } catch (_) { /* quota: play on without saving */ }
  }
}

async function fetchText(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(path + ": " + res.status);
  return res.text();
}

async function boot() {
  const [mod, seedLog, triggers] = await Promise.all([
    WebAssembly.instantiateStreaming(fetch("munou_web.wasm")).catch(async () => {
      const buf = await (await fetch("munou_web.wasm")).arrayBuffer();
      return WebAssembly.instantiate(buf);
    }),
    fetchText("seed.jsonl"),
    fetchText("triggers.example.json").catch(() => null),
  ]);
  wasm = mod.instance.exports;

  const savedSeed = localStorage.getItem(SEED_KEY);
  if (savedSeed !== null) seedBox.value = savedSeed;
  const savedLog = localStorage.getItem(LOG_KEY);

  wasm.set_now(Date.now());
  put(savedLog !== null ? savedLog : seedLog);
  if (wasm.init(Number(seedBox.value) || 1) !== 0) {
    add("note", "起動に失敗: " + out());
    return;
  }
  if (triggers) {
    put(triggers);
    wasm.load_triggers();
  }
  input.disabled = false;
  send.disabled = false;
  input.focus();
  add("note", savedLog !== null
    ? "前回のつづきから（この端末のログを再生して同じ状態に戻りました）"
    : "はじめまして（種ログ50発話から）");
}

document.getElementById("form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const text = input.value.trim();
  if (!text || !wasm) return;
  input.value = "";
  add("user", text);
  wasm.set_now(Date.now());
  put(text);
  if (wasm.respond() !== 0) {
    add("note", "エラー: " + out());
    return;
  }
  const r = JSON.parse(out());
  if (r.interject) add("bot", r.interject);
  add("bot", r.text);
  if (r.milestone) add("note", "（" + r.milestone + "）");
  add("strip", r.strip);
  lastWhy = r.why;
  persist();
});

let lastWhy = null;

function panelButton(id, call) {
  document.getElementById(id).addEventListener("click", () => {
    if (!wasm) return;
    if (call() === 0) addPanel(out());
  });
}
panelButton("btn-observe", () => wasm.observe());
panelButton("btn-ayumi", () => { wasm.set_now(Date.now()); return wasm.ayumi(); });
document.getElementById("btn-why").addEventListener("click", () => {
  if (!wasm) return;
  if (lastWhy) addPanel(lastWhy);
  else if (wasm.why() === 0) addPanel(out());
});
document.getElementById("btn-good").addEventListener("click", () => {
  if (wasm && wasm.feedback(1) === 0) { add("note", out()); persist(); }
});
document.getElementById("btn-bad").addEventListener("click", () => {
  if (wasm && wasm.feedback(0) === 0) { add("note", out()); persist(); }
});
document.getElementById("btn-reset").addEventListener("click", () => {
  if (!confirm("ログを消して、種ログから育て直します。いいですか？")) return;
  localStorage.removeItem(LOG_KEY);
  localStorage.setItem(SEED_KEY, seedBox.value);
  location.reload();
});
seedBox.addEventListener("change", () => localStorage.setItem(SEED_KEY, seedBox.value));

boot().catch((e) => add("note", "読み込みに失敗しました: " + e));
