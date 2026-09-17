// Raw-ABI WASM bridge. Recreate memory views after every call: growth detaches them.
"use strict";

const LOG_KEY = "munou.log";
const SEED_KEY = "munou.seed";
const enc = new TextEncoder();
const dec = new TextDecoder();
// Keep the script and its WASM data contract on the same cache revision.
const WASM_URL = "munou_web.wasm" + new URL(document.currentScript.src).search;
let wasm = null;
let ready = false;

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
const engineButtons = ["btn-observe", "btn-history", "btn-why", "btn-good", "btn-bad"]
  .map((id) => document.getElementById(id));
engineButtons.forEach((button) => { button.disabled = true; });

const pathLabels = {
  trigger: "定型の返答",
  markov: "言葉の組み立て",
  retrieve: "過去の返答",
  echo: "入力の繰り返し",
  adapt: "過去のやり取りから",
};
const stageLabels = {
  empty: "まだ会話の記録がありません",
  logged: "会話を記録しています",
  sprout: "言葉を覚えはじめました",
  growing: "言葉が増えてきました",
  dense: "会話の記録が増えてきました",
};
const number = new Intl.NumberFormat("ja-JP");
const date = new Intl.DateTimeFormat("ja-JP", { year: "numeric", month: "long", day: "numeric", timeZone: "UTC" });
const count = (value) => number.format(value);
const percent = (value) => `${Math.round(value)}%`;
const pathName = (path) => pathLabels[path] || "記録なし";

function element(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function message(cls, text) {
  return element("p", `msg ${cls}`, text);
}

// Append a whole turn together, then scroll only once after layout changes.
function append(...nodes) {
  const fragment = document.createDocumentFragment();
  nodes.forEach((node) => fragment.appendChild(node));
  chat.appendChild(fragment);
  const last = nodes[nodes.length - 1];
  last?.scrollIntoView({ block: last.matches(".panel, .observation.full") ? "start" : "end" });
}

function panel(title) {
  const node = element("section", "panel");
  node.appendChild(element("h2", "panel-title", title));
  return node;
}

function facts(rows) {
  const list = element("dl", "facts");
  for (const [label, value] of rows) {
    list.append(element("dt", "", label), element("dd", "", value));
  }
  return list;
}

function textList(title, values, emptyText) {
  const section = element("div", "record-section");
  section.appendChild(element("h3", "", title));
  if (!values.length) {
    section.appendChild(element("p", "panel-note", emptyText));
  } else {
    const list = element("ul", "record-list");
    values.forEach((value) => list.appendChild(element("li", "", value)));
    section.appendChild(list);
  }
  return section;
}

function metric(label, value, fraction, description) {
  const node = element("div", "metric");
  node.append(element("span", "metric-label", label), element("span", "metric-value", value));
  const track = element("div", "metric-track");
  const fill = element("div", "metric-fill");
  const width = fraction === null ? 0 : Math.max(0, Math.min(1, fraction)) * 100;
  fill.style.width = `${width}%`;
  track.appendChild(fill);
  if (fraction !== null) {
    track.setAttribute("role", "meter");
    track.setAttribute("aria-label", label);
    track.setAttribute("aria-valuemin", "0");
    track.setAttribute("aria-valuemax", "100");
    track.setAttribute("aria-valuenow", String(width));
    track.setAttribute("aria-valuetext", value);
  } else {
    track.setAttribute("aria-hidden", "true");
  }
  node.append(track, element("p", "metric-description", description));
  return node;
}

function observation(o, compact = false) {
  const node = element("section", `observation ${compact ? "compact" : "full"}`);
  const heading = element("header", "observation-head");
  heading.append(
    element("h2", "panel-title", "会話のようす"),
    element("span", "observation-stage", stageLabels[o.stage]),
  );
  node.appendChild(heading);
  const measured = o.eval_n > 0;
  const novelty = Math.max(0, Math.min(1, 1 - o.rote_lcs));
  const metrics = element("div", "metrics");
  metrics.append(
    metric("学習に使った発言", percent(o.absorb_rate * 100), o.absorb_rate,
      `会話の${count(o.utterances)}件中、${count(o.learned)}件を返答の材料に取り込んでいます。`),
    metric("言葉の種類", `${count(o.vocab)}種類`, o.vocab / 200,
      "単語や短いまとまりを数えています。目盛りは200種類までですが、その先も増えます。"),
    metric("話題とのつながり", measured ? percent(o.band_hit_pct) : "記録なし", measured ? o.band_hit_pct / 100 : null,
      `話題との近さが設定範囲（${o.band_lo.toFixed(2)}〜${o.band_hi.toFixed(2)}）に収まった返答の割合です。内容の正しさを示す値ではありません。`),
    metric("返答の新しさ", measured ? percent(novelty * 100) : "記録なし", measured ? novelty : null,
      "過去の返答と同じ言葉の並びが少ないほど高く表示します。意味の新しさではありません。"),
    metric("変化をつけた返答", measured ? percent(o.slip_pct) : "記録なし", measured ? o.slip_pct / 100 : null,
      "最も点数の高い候補以外を選んだ返答の割合です。"),
  );
  node.appendChild(metrics);
  if (!compact) {
    node.appendChild(facts([
      ["会話の記録", `${count(o.utterances)}件`],
      ["学習した言葉の総数", `${count(o.tokens)}個（区切りを含む）`],
      ["評価した返答", `${count(o.eval_n)}件`],
      ["直前の返し方", pathName(o.last_path)],
    ]));
    const paths = Object.entries(o.paths).filter(([, n]) => n > 0)
      .map(([path, n]) => `${pathName(path)}：${count(n)}件`);
    node.appendChild(textList("返し方の内訳", paths, "まだ返し方の記録がありません。"));
    node.appendChild(textList("最近、学習に使った返答", o.recent_learned_bot, "まだありません。"));
  }
  return node;
}

function explanation(data) {
  const node = panel("返答の理由");
  const trace = data.trace;
  if (!trace) {
    const o = data.observation;
    if (o.last_path) {
      node.appendChild(facts([["直前の返し方", pathName(o.last_path)]]));
      node.appendChild(element("p", "panel-note", "再読み込み前の候補の詳細は残っていません。次の返答から確認できます。"));
    } else {
      node.appendChild(element("p", "panel-note", "まだ返答の詳しい記録がありません。まずは話しかけてみてください。"));
    }
    return node;
  }
  node.appendChild(element("p", "panel-note", "話題との近さや繰り返しの少なさなどで候補を比べ、返答を選んでいます。"));
  node.appendChild(facts([
    ["入力", trace.input],
    ["返し方", pathName(trace.path)],
    ["学習への取り込み", trace.learned ? "今回のやり取りを学習に使いました" : "今回は記録のみ残しました"],
  ]));
  if (trace.slipped) {
    node.appendChild(element("p", "panel-note", "今回は変化をつけるため、最高得点以外の候補を選びました。"));
  }
  for (const candidate of trace.candidates) {
    const row = element("div", `candidate${candidate.chosen ? " chosen" : ""}`);
    row.append(
      element("strong", "", candidate.chosen ? "選ばれた返答" : `候補 ${candidate.rank + 1}`),
      element("p", "", candidate.text),
      element("p", "panel-note", `${pathName(candidate.source)} · 得点 ${candidate.score.toFixed(3)} · 話題との近さ ${candidate.topic_score.toFixed(3)}`),
    );
    node.appendChild(row);
  }
  return node;
}

function historyPanel(data) {
  const h = data.history;
  const o = data.observation;
  const node = panel("これまでの記録");
  const rows = [
    ["記録開始", h.first_speech_t === null ? "まだ記録がありません" : date.format(h.first_speech_t)],
    ["記録の期間", h.first_speech_t === null ? "—" : `${count(h.age_days + 1)}日`],
    ["最初に覚えた言葉", h.first_learned_user ?? "まだありません"],
    ["会話の記録", `${count(o.utterances)}件`],
    ["言葉の種類", `${count(o.vocab)}種類`],
  ];
  if (h.weather) {
    const labels = { "なぎ": "いつもどおり", "はずみ": "相づちが多め", "しめり": "控えめ", "きまぐれ": "変化が多め" };
    rows.push(["返答の調子", labels[h.weather] || h.weather]);
  }
  if (h.care_word) rows.push(["今日、使いやすい言葉", h.care_word]);
  node.appendChild(facts(rows));
  if (h.aloof) node.appendChild(element("p", "panel-note", "前の会話から間が空いたため、しばらくは控えめに返します。"));
  node.appendChild(textList("よく出てくる言葉", h.interests.map((item) => item.word), "まだ十分な記録がありません。"));
  node.appendChild(textList("これまでの節目", [
    ...h.learned_marks.map((n) => `${count(n)}件の発言を学習に使用`),
    ...h.day_marks.map((n) => `記録開始から${count(n)}日`),
  ], "これから会話を重ねていきましょう。"));
  return node;
}

function milestoneText(text) {
  // These messages are engine-generated milestones, never conversation text.
  return text.replace(/^節目 吸収(\d+)$/, "学習に使った発言が$1件になりました。")
    .replace(/^節目 (\d+)日目$/, "記録を始めて$1日が経ちました。");
}

function persist() {
  if (wasm.export_log() === 0) {
    try { localStorage.setItem(LOG_KEY, out()); } catch (_) { /* Keep the active conversation if storage is full. */ }
  }
}

async function fetchText(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(path + ": " + res.status);
  return res.text();
}

async function loadWasm() {
  const response = await fetch(WASM_URL);
  if (!response.ok) throw new Error("プログラムを読み込めませんでした: " + response.status);
  try {
    return await WebAssembly.instantiateStreaming(response.clone());
  } catch (_) {
    return WebAssembly.instantiate(await response.arrayBuffer());
  }
}

function selectedSeed() {
  const value = Number(seedBox.value || "1");
  return Number.isSafeInteger(value) && value >= 0 ? value : 1;
}

async function boot() {
  const [mod, seedLog, triggers] = await Promise.all([
    loadWasm(), fetchText("seed.jsonl"), fetchText("triggers.example.json").catch(() => null),
  ]);
  wasm = mod.instance.exports;
  const savedSeed = localStorage.getItem(SEED_KEY);
  if (savedSeed !== null) seedBox.value = savedSeed;
  const savedLog = localStorage.getItem(LOG_KEY);
  wasm.set_now(Date.now());
  put(savedLog !== null ? savedLog : seedLog);
  if (wasm.init(selectedSeed()) !== 0) {
    append(message("note", "会話を始められませんでした: " + out()));
    return;
  }
  if (triggers) {
    put(triggers);
    wasm.load_triggers();
  }
  ready = true;
  input.disabled = false;
  send.disabled = false;
  engineButtons.forEach((button) => { button.disabled = false; });
  append(message("note", savedLog !== null
    ? "このブラウザに保存した会話を読み込みました。"
    : "短い会話の記録を用意しています。まずは話しかけてみてください。"));
  input.focus({ preventScroll: true });
}

document.getElementById("form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const text = input.value.trim();
  if (!text || !ready) return;
  input.value = "";
  const nodes = [message("user", text)];
  wasm.set_now(Date.now());
  put(text);
  if (wasm.respond() !== 0) {
    append(...nodes, message("note", "返答を作れませんでした: " + out()));
    return;
  }
  const r = JSON.parse(out());
  if (r.interject) nodes.push(message("bot", r.interject));
  nodes.push(message("bot", r.text));
  if (r.milestone) nodes.push(message("note", milestoneText(r.milestone)));
  nodes.push(observation(r.observation, true));
  persist();
  append(...nodes);
  input.focus({ preventScroll: true });
});

function panelButton(id, call, render) {
  document.getElementById(id).addEventListener("click", () => {
    if (!ready) return;
    if (call() === 0) append(render(JSON.parse(out())));
    else append(message("note", "表示できませんでした: " + out()));
  });
}
panelButton("btn-observe", () => wasm.observe(), (data) => observation(data));
panelButton("btn-history", () => { wasm.set_now(Date.now()); return wasm.history(); }, historyPanel);
panelButton("btn-why", () => wasm.why(), explanation);

for (const [id, good] of [["btn-good", 1], ["btn-bad", 0]]) {
  document.getElementById(id).addEventListener("click", () => {
    if (!ready) return;
    if (wasm.feedback(good) === 0) {
      const note = out();
      persist();
      append(message("note", note));
    } else append(message("note", "記録できませんでした: " + out()));
  });
}
document.getElementById("btn-reset").addEventListener("click", () => {
  if (!confirm("このブラウザに保存した会話を消して、最初の状態に戻します。よろしいですか？")) return;
  localStorage.removeItem(LOG_KEY);
  localStorage.setItem(SEED_KEY, String(selectedSeed()));
  location.reload();
});
seedBox.addEventListener("change", () => localStorage.setItem(SEED_KEY, String(selectedSeed())));
boot().catch((error) => append(message("note", "読み込めませんでした。ページを再読み込みしてください。詳細: " + error.message)));
