# 人工無脳君 v0.1 検証記録

日付: 2026-08-19  
環境: rustc 1.83.0, Linux x86_64, release (`lto=true`, `codegen-units=1`)  
再現: `cargo run -p munou-cli --release -- verify --sa-tokens 10000000 --turns 300`

この文書は設計書 draft v0.1 に対する実装の突き合わせである。数値は上記コマンドの実測。

## 結論

v0.1 の約束（CLI、閉じたマルコフ生成、説明可能な選択、会話ログから育つ）は満たしている。v0.1.1 で候補は XOR ではなくプール（トリガー / 検索 / マルコフ / エコー）。設計 §2.2 の速度・常駐メモリは **release 実測で要件内**。mmap コールドスタートとホットパスのヒープゼロは未実装で、検証コマンドでは SKIP としている。

検証中に直したもの:

- マルコフ文脈に **現在のユーザーチャンクが入っていなかった**（履歴だけを渡していた）
- トークナイズが形態素/チャンクの `String` を二重に保持して **debug で 50MB/s を大幅に下回った**。スライス intern に変更
- CI の `dtolnay/rust-toolchain@1.83.0` が解決できない。`@master` + `toolchain: 1.83.0` に変更

## 自動テスト

| 項目 | 結果 |
|---|---|
| `cargo fmt --all -- --check` | PASS |
| `cargo clippy --workspace --all-targets -- -D warnings` | PASS |
| `cargo test --workspace` | PASS（unit 24 + spec 13） |

spec テストはトリガー排他、`p_slip`、説明チェーン、JSONL がソース・オブ・トゥルース、KN 応答、ユーザー文脈、lockfile の閉じた依存、`sync_data` 後の再開を固定する。

## 四性質

| 性質 | 判定 | 根拠 |
|---|---|---|
| 説明可能 | PASS | `/why` と `say --explain` が path・形態素・チャンク・候補スコア・ズレ roll・生成ステップを出す。spec `explain_chain_is_complete` |
| 育てられる | PASS | `log.jsonl` が追記され再オープンで復元。`data/seed.jsonl` 25往復で tokens 0→176 / vocab 0→99。同じ8入力で空と 7/8 が食い違う（`munou probe`）。候補はトリガー/検索/マルコフ/エコーのプール |
| 閉じている | PASS | `Cargo.lock` に reqwest / candle / tokenizers / llama 等なし。生成はマルコフのみ。embedding は選択用ハッシュ。学習コーパスは自分のログ。トリガー辞書はユーザー供給（形態素辞書と同じ例外枠） |
| ズレる | PASS | `p_slip=0` で slip なし、`p_slip=1` かつ候補≥2 で 2 位以下を採用 |

## パイプライン（CLI スモーク）

`data/triggers.example.json`、seed=1:

1. `おはよう` → path=Trigger、辞書応答 `おはよう`、sim=1.000
2. `今日はいい天気だね` → path=Markov。コーパスが挨拶中心なので候補も挨拶に寄る（閉じていることの実演）
3. `log.jsonl` に user/bot が 2 ターン分追記され、`score` / `slipped` が残る
4. 別ディレクトリ・同一シードで `散歩しようか` の応答が一致

## 非機能要件 §2.2

| 項目 | 要件 | 実測 | 判定 |
|---|---|---|---|
| 応答レイテンシ（embed 込み） | p99 ≤ 30ms | 2.84ms（300 turn） | PASS |
| エンジン部（マルコフ 1 本） | p99 ≤ 2ms | 77µs | PASS |
| トークナイズ | ≥ 50MB/s 単スレッド | 75.8 MB/s | PASS |
| SA 全再構築 | 10^7 トークン ≤ 2s | 1.023s | PASS |
| 常駐メモリ | 10^7 トークン時 ≤ 256MB | ロード後 VmRSS 105MB | PASS |
| （参考）SA-IS 構築ピーク | 予算なし | VmRSS 355MB | 記録のみ |
| 起動 | コールド ≤ 100ms（mmap 遅延ロード） | 空ログは ≪100ms。インデックスは JSONL から再構築 | 空ログ PASS / mmap SKIP |
| 再現性 | 同一ログ+同一シード | エンジン・CLI とも一致。slip も ChaCha8 | PASS |
| 依存 | 単一バイナリ、外部サービスなし | lockfile 検査 | PASS |
| 耐障害 | append-only + fsync | `sync_data` のあと別 `Engine::open` で復元 | PASS |
| ホットパスヒープゼロ | 目標 | `respond` が Vec/String を確保 | SKIP |

debug ビルドのトークナイズは ~6MB/s で要件未満、engine-p99 も debug では 2ms 際どいため、`munou verify` は debug で外れた tokenize / latency-p99 / engine-p99 を SKIP する。CI の軽量 verify も debug。NFR の正は release。

## コンポーネント対応

| 設計 | v0.1 | 判定 |
|---|---|---|
| トークナイザー ゼロ設計 | 分岐エントロピー + AV。弱モデル時 CJK 文字単位。統計チャンク | PASS（文節ではない、設計どおり） |
| intern u32 | `Interner` | PASS |
| コーパス = u32 列 + SA-IS | `Store.text` + `Store.sa`、バッファ世代マージ | PASS |
| 可変長 n-gram | SA 二分探索 + 最長一致バックオフ | PASS |
| 素朴バックオフ既定、KN は trait | `Smoothing`、`--smoothing kn` | PASS |
| alias 法 + 温度 | Vose alias、`τ_gen` | PASS |
| 選択 = embed ランク + p_slip | ハッシュ embed dim 256、2 位以下を加重サンプル | PASS |
| 話題ベクトル MA | `TopicTracker` k=5 | PASS |
| トリガー閾値ゆるめ | θ=0.42、余弦 | PASS |
| 対話行為分類器 | なし | 後段（設計どおり省略可） |
| 固有記憶 KV | なし | v2 |
| i8 量子化 / SIMD / ANN | なし | 楽しみ枠 |
| mmap インデックス | なし | 未実装 |

## パラメータ初期値

実装の `Params::default` は設計 §5 の目安と一致: N_cand=10, τ=1.0, L_max=8, f_min=3, k_topic=5, p_slip=0.15。θ_trig は 0.42 で「意図的に緩く」。

## 評価 §6

面白さの目的関数は未解決のまま。`/eval` は帯域ヒット率 `[0.25, 0.85]` と既存発話との token LCS だけ。トリガー完全一致は sim=1.0 で帯域外（band_hit=false）になり、これは「高すぎる類似は面白くない」という代理指標の意図に合う。

## シードログ数値（`munou probe`）

`data/seed.jsonl` は散歩・コーヒー・猫・仕事・ゲームを繰り返した 25往復（50レコード）。学習はこれだけ。外部コーパスは使わない。v0.1.1 から既定は候補プール（`--mix pool`）。

再現: `cargo run -p munou-cli --release -- probe --seed data/seed.jsonl`（rng=1, p_slip=0, `data/triggers.example.json`）

| 項目 | 空エンジン | シード後 | 読み |
|---|---|---|---|
| utterances | 0 | 50 | ログがコーパス |
| tokens | 0 | 176 | SA に載るチャンク数 |
| vocab | 0 | 99 | intern した表層 |
| 8入力のうち応答が違う | — | 7/8 (88%) | ログが生成を変える |
| trigger 率 | 2/8 | 2/8 | `おはよう` / `ありがとう` は辞書が勝つ |
| 勝ち筋 | Echo/Mark 混在 | Trigger / Retrieve / Markov | 同じ入力でもソースが分かれる |
| mean ctx_len_used | 0.38 | 0.50 | 最長一致がわずかに伸びる |
| mean_sim | 0.474 | 0.381 | 帯域 `[0.25,0.85]` の内側寄り |
| band_hit | — | 88% | 8本中7本 |
| rote_lcs | — | 0.75 | 検索が勝つと既存発話に寄る |
| slip | — | 0% | p_slip=0 |
| 応答 mean / max | — | 174µs / 248µs | release、embed 込み |
| hybrid-pool | — | 最大3ソース | ドメイン内プロンプトの候補リスト |

ドメイン外の `量子力学の話をしよう` は Trigger に落ちず Markov `うちの猫かわいい散歩しない？？`。知っていることはログ由来。空側のマルコフはユーザー文の破片、シード側はログ n-gram の組み換えか検索。

`munou probe` は tokens/vocab 増加、≥3本の食い違い、おはよう=Trigger、OOD≠Trigger、プールに複数ソース、決定性、p_slip=0 で slip なしを PASS/FAIL する。CI もこれを回す。

## 検証コマンドの読み方

```
PASS / FAIL  … 必須。FAIL があるとプロセスは非ゼロ終了
SKIP         … v0.1 でやらないと決めた項目、または debug では測らない NFR
```

`--sa-tokens` が 10^7 未満だと `rss-1e7` は SKIP。CI は `verify --sa-tokens 20000 --turns 40` と `probe --seed data/seed.jsonl`。