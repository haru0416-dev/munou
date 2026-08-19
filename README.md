# 人工無脳君 (`munou`)

LLM を使わず、理解せずに会話が成立する対話プログラム。賢さではなく次の四つを目標にする。

1. **説明可能** — 応答は生成チェーンと選択スコアで追跡できる (`/why`)
2. **育てられる** — 語彙・遷移・記憶はユーザー可視。会話ログから育つ
3. **閉じている** — 外部知識なし。知っていることは自分の会話ログ由来
4. **ズレる** — 文脈からの適度な逸脱は仕様 (`p_slip`)

生成は可変長マルコフ（接尾辞配列の上の最長一致バックオフ）、検索は過去の自分の発話、選択は閉じたハッシュ embedding。生成用言語モデルは無い。既定はこれらの候補をプールして一本を選ぶ（`--mix exclusive` で v0.1 の XOR に戻せる）。

```bash
cargo run -p munou-cli --release -- chat --seed 1 --triggers data/triggers.example.json
```

```
人工無脳君  seed=1  /why /stats /eval /rebuild /retok /explain /quit
> おはよう
おはよう
> /why
path=Trigger  elapsed=141us  slipped=false (p_slip=0.15 roll=0.402)
...
```

REPL: `/why` トレース、`/stats` コーパス、`/eval` 帯域ヒットと丸暗記 LCS、`/rebuild` SA 再構築、`/retok` ログ全体を再分割。

同一ログを空から同じシードで再生すると応答列は完全一致する。

```bash
cargo run -p munou-cli --release -- bench --tokens 10000000
# SA-IS n=10000000  ~1.0s   (要件: ≤ 2s)

cargo run -p munou-cli --release -- verify --sa-tokens 10000000 --turns 300
cargo run -p munou-cli --release -- probe --seed data/seed.jsonl
```

`probe` は `data/seed.jsonl`（25往復の雑談ログ）を読み、空エンジンと育てたエンジンで同じ入力を投げて数値を出す。

設計書との突き合わせは [`docs/verify.md`](docs/verify.md)。

クレート構成:

- `munou-engine` — インターフェース非依存のコア
- `munou-cli` — 最初のアダプタ。bot / 常駐は後から足す

設計の本文と v0.1 での決定は [`docs/design.md`](docs/design.md)。
