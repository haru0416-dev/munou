# 人工無脳君 (`munou`)

LLM を使わず、理解せずに会話が成立する対話プログラム。賢さではなく次の四つを目標にする。製品の本線は会話そのものより、**育つ様子を観察すること**。

1. **説明可能** — 応答は生成チェーンと選択スコアで追跡できる (`/why`)
2. **育てられる** — 語彙・遷移・記憶はユーザー可視。会話ログは毎回残り、コーパスへの吸収は対話中にランダム (`p_learn`)。ゲージは `munou observe` / `/observe`
3. **閉じている** — 外部知識なし。知っていることは自分の会話ログ由来
4. **ズレる** — 文脈からの適度な逸脱は仕様 (`p_slip`)

生成は可変長マルコフ（接尾辞配列の上で次数を補間）、検索は MMR、選択は閉じたハッシュ embedding に帯域ヒンジ。生成用言語モデルは無い。既定はこれらの候補をプールして一本を選ぶ（`--mix exclusive` で v0.1 の XOR に戻せる）。会話ログは毎回残し、コーパス（SA・検索・トークナイザ）への吸収は対話中にランダム（`--p-learn`、既定 0.35）。

観察窓のゲージは既存指標だけ（吸収率・語彙・帯域ヒット・1−rote・slip）。感情モデルは足さない。stage（空 / 記録中 / 芽生え / 育ち / 濃い）もカウントから決める。記憶は作業（話題窓）・挿話（ログ）・母数（SA / 語彙）として同じ数字を並べるだけ。

LLM は使わない。HuggingFace やチャット API も使わない。温度・nucleus・経路ゲート・好みラベルは、閉じたマルコフと選択器の上に**概念だけ**写したもの。

**アダプタ（伺か SHIORI / Misskey / 常駐）は後段。** いまは CLI だけ。

```bash
cargo run -p munou-cli --release -- chat --seed 1 --triggers data/triggers.example.json
cargo run -p munou-cli --release -- observe --data-dir ./munou-data
cargo run -p munou-cli --release -- observe --data-dir ./munou-data --format html > observe.html
```

```
人工無脳君  seed=1  /observe /why /good /bad /stats /eval /rebuild /retok /explain /quit
> おはよう
おはよう
観察 芽生え  吸収██████ 100%  語彙█░░░░░   3  帯域░░░░░░    -  暗記░░░░░░    -  ズレ░░░░░░   -  echo
> /observe
人工無脳君 観察窓  stage=芽生え
...
```

REPL: 応答の直後に一行ゲージ。`/observe` が本線、`/why` トレース、`/good` `/bad` が直前の経路を少し好き/嫌いにする（コーパスには入らない）、`/stats` コーパス、`/eval` 帯域ヒットと丸暗記 LCS、`/rebuild` SA 再構築、`/retok` ログ全体を再分割。`--top-p` は nucleus（既定オフ）。`--format html` はローカルの自己完結 HTML（サーバではない）。

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
- `munou-cli` — 観察窓と REPL。伺か / Misskey / 常駐は後段

設計の本文と v0.1 での決定は [`docs/design.md`](docs/design.md)。
