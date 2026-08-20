# 人工無脳君 v0.1 検証記録

日付: 2026-08-19  
環境: rustc 1.83.0, Linux x86_64, release (`lto=true`, `codegen-units=1`)  
再現: `cargo run -p munou-cli --release -- verify --sa-tokens 10000000 --turns 300`

この文書は設計書 draft v0.1 に対する実装の突き合わせである。数値は上記コマンドの実測。

## 結論

v0.1 の約束（CLI、閉じたマルコフ生成、説明可能な選択、会話ログから育つ）は満たしている。v0.1.1 で候補は XOR ではなくプール。v0.1.2 でライブ吸収は `p_learn`。v0.1.3 で観察窓が本線（アダプタは劣後）。v0.1.4 で次数補間・Witten-Bell・帯域ヒンジ・Boltzmann slip・連続部分列の丸暗記検出。v0.1.5 で LLM エコシステムの概念だけを閉じた構造へ写す（nucleus・経路ゲート・記憶三層・`/good` `/bad`）。v0.1.6 で閉じた雑談ログを大量複製（`munou scale`）。v0.1.7 で modified KN / PPM 除外 / skip-gram / recency cache を SA マルコフへ写す（KenLM は使わない）。v0.1.8 で育ったログの応答時間を計測に加え、疎表現で予算内へ（応答列は変わる。分布同一はテストで固定）。API も重みも使っていない。設計 §2.2 の速度・常駐メモリは **release 実測で要件内**。mmap コールドスタートとホットパスのヒープゼロは未実装で、検証コマンドでは SKIP としている。

検証中に直したもの:

- マルコフ文脈に **現在のユーザーチャンクが入っていなかった**（履歴だけを渡していた）
- トークナイズが形態素/チャンクの `String` を二重に保持して **debug で 50MB/s を大幅に下回った**。スライス intern に変更
- CI の `dtolnay/rust-toolchain@1.83.0` が解決できない。`@master` + `toolchain: 1.83.0` に変更

## 自動テスト

| 項目 | 結果 |
|---|---|
| `cargo fmt --all -- --check` | PASS |
| `cargo clippy --workspace --all-targets -- -D warnings` | PASS |
| `cargo test --workspace` | PASS（unit 45 + spec 23 + soak 16） |

spec テストはトリガー排他、`p_slip`、説明チェーン、JSONL がソース・オブ・トゥルース、KN 応答、skip/cache/PPM フラグ、ユーザー文脈、lockfile の閉じた依存、`sync_data` 後の再開、`p_learn` の吸収/スキップ、観察窓（空 / シード育ち / 記録中 / path 再開）、meta 好みがコーパスに入らないこと、経路ゲート後もエコーがプールに残ることを固定する。soak は変な入力・壊れた JSONL・パラメータ端・再開後の `/good` `/why`・80ターンのレイテンシを回す。

## 四性質

| 性質 | 判定 | 根拠 |
|---|---|---|
| 説明可能 | PASS | `/why` と `say --explain` が path・形態素・チャンク・候補スコア・ズレ roll・生成ステップを出す。spec `explain_chain_is_complete` |
| 育てられる | PASS | `log.jsonl` が追記され再オープンで復元。ライブでは `p_learn` でコーパスへ吸収（既定 0.35）。シードログは `learned` 省略＝吸収済み。`data/seed.jsonl` 25往復で tokens 0→176 / vocab 0→99。同じ8入力で空と 7/8 が食い違う（`munou probe`）。候補はトリガー/検索/マルコフ/エコーのプール。観察窓は発話・吸収・語彙・帯域・暗記・ズレのゲージ（`munou observe`）。stage はカウント由来。伺か / Misskey は未実装（劣後） |
| 閉じている | PASS | `Cargo.lock` に reqwest / candle / tokenizers / llama 等なし。生成はマルコフのみ。embedding は選択用ハッシュ。学習コーパスは自分のログ。トリガー辞書はユーザー供給（形態素辞書と同じ例外枠） |
| ズレる | PASS | `p_slip=0` で slip なし、`p_slip=1` かつ候補≥2 で 2 位以下を採用 |

## パイプライン（CLI スモーク）

`data/triggers.example.json`、seed=1:

1. `おはよう` → path=Trigger、辞書応答 `おはよう`、sim=1.000
2. `今日はいい天気だね` → path=Markov。コーパスが挨拶中心なので候補も挨拶に寄る（閉じていることの実演）
3. `log.jsonl` に user/bot が 2 ターン分追記され、`score` / `slipped` / `learned` / `path` が残る
4. 別ディレクトリ・同一シードで `散歩しようか` の応答が一致
5. `munou observe` が非空のパネルを出し、シードログでは stage=育ち・learned=50

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
| 素朴バックオフ既定、modified KN は trait | `Smoothing`、`--smoothing kn`。KenLM なし | PASS |
| alias 法 + 温度 | Vose alias、`τ_gen` | PASS |
| 選択 = embed ランク + p_slip | ハッシュ embed dim 256、2 位以下を加重サンプル | PASS |
| 話題ベクトル MA | `TopicTracker` k=5 | PASS |
| トリガー閾値ゆるめ | θ=0.42、余弦 | PASS |
| 対話行為分類器 | なし | 後段（設計どおり省略可） |
| 固有記憶 KV | なし | v2 |
| i8 量子化 / SIMD / ANN | なし | 楽しみ枠 |
| mmap インデックス | なし | 未実装 |

## パラメータ初期値

実装の `Params::default` は設計 §5 の目安と一致: N_cand=10, τ=1.0, L_max=8, f_min=3, k_topic=5, p_slip=0.15, p_learn=0.35。θ_trig は 0.42 で「意図的に緩く」。

## 評価 §6

面白さの目的関数は未解決のまま。`/eval` は帯域ヒット率 `[0.25, 0.85]` と既存発話との **連続部分列**。選択器も同じ帯域にヒンジをかけ、余弦を無制限に最大化しない。トリガー完全一致は sim=1.0 で帯域外（band_hit=false）になり、これは「高すぎる類似は面白くない」という代理指標の意図に合う。

## シードログ数値（`munou probe`）

`data/seed.jsonl` は散歩・コーヒー・猫・仕事・ゲームを繰り返した 25往復（50レコード）。学習はこれだけ。外部コーパスは使わない。v0.1.1 から既定は候補プール（`--mix pool`）。

再現: `cargo run -p munou-cli --release -- probe --seed data/seed.jsonl`（rng=1, p_slip=0, p_learn=1, `data/triggers.example.json`）

| 項目 | 空エンジン | シード後 | 読み |
|---|---|---|---|
| utterances | 0 | 50 | ログがコーパス |
| tokens | 0 | 176 | SA に載るチャンク数 |
| vocab | 0 | 99 | intern した表層 |
| 8入力のうち応答が違う | — | 7/8 (88%) | ログが生成を変える |
| trigger 率 | 2/8 | 2/8 | `おはよう` / `ありがとう` は辞書が勝つ |
| 勝ち筋 | Echo/Mark 混在 | Trigger / Retrieve / Markov | 同じ入力でもソースが分かれる |
| mean ctx_len_used (Markov) | 0.50 | 0.33 | 勝ち筋が検索に寄ると Markov 行が減る。ゲートの意図 |
| mean_sim | 0.447 | 0.359 | 帯域ヒンジで高すぎる類似を抑える |
| band_hit | — | 75% | 8本中6本 |
| rote_lcs | — | 0.75 | 連続部分列。検索が勝つと既存発話に寄る |
| slip | — | 0% | p_slip=0 |
| 応答 mean / max | — | 183µs / 294µs | release、embed 込み |
| hybrid-pool | — | 最大3ソース | ドメイン内プロンプトの候補リスト |

ドメイン外の `量子力学の話をしよう` は Trigger に落ちない。知っていることはログ由来。空側のマルコフはユーザー文の破片、シード側はログ n-gram の組み換えか検索。ライブでは `p_learn`（probe は 1）でコーパスへ吸収し、会話ログは毎回残す。

`munou probe` は tokens/vocab 増加、≥3本の食い違い、おはよう=Trigger、OOD≠Trigger、プールに複数ソース、決定性、p_slip=0 で slip なしを PASS/FAIL する。CI もこれを回す。

## 巨大ログ（`munou scale`、release、2026-08-19）

外部コーパスは使っていない。シードと同じ雑談型の組み合わせ。`data/seed.jsonl` は 50 のまま。

| pairs | 発話 | tokens | vocab | open | respond p99 | 読み |
|---|---|---|---|---|---|---|
| 25（seed） | 50 | 176 | 99 | ≪1ms | ~0.3ms | 育ち |
| 10_000 | 20_000 | 69_468 | 103 | 65ms | 2.4ms | 語彙はほぼ固定。濃い |
| 50_000 | 100_000 | 347_989 | 103 | 737ms | 7.4ms | 30ms 予算内。検索は直近 1024 |
| 20_000 unique_frac=0.15 | 40_000 | 140_668 | 7_069 | 210ms | 38ms | 表層を増やすと retrieve が重い |

`--unique-frac 0` が「同じ人と長く話した」。語彙を増やすと p99 が 30ms を超えることがある。

## 育ったログの応答時間（v0.1.8、2026-08-19）

上の表の「語彙を増やすと p99 が 30ms を超えることがある」を潰した記録。環境: rustc 1.96.1、このリポジトリの CI と同じ 6 コア VPS、release。

### 限界を先に書く

- 計測に使ったログは `fabricate`（シード話題の組み換え + `unique_frac` による表層追加）で、実ユーザーのログではない。実ログの語彙分布では未検証
- 各シナリオ n=60 ターン（12 プロンプト × 5 周）、1 台のマシン。他環境では測っていない
- `Engine::open` の時間は予算を設けておらず、今も発話数に比例する（10 万発話で ~0.87s）
- v0.1.8 は**応答列が v0.1.7 と一致しない**（後述の宣言部分）。同一バイナリ内の再現性（同一ログ + 同一シード → 同一応答列）は保っている

### 計測が見ていなかったもの

v0.1.7 までの `latency-p99` は**空のエンジン**（ウォームアップ 4 発 + テスト中の吸収のみ）で測っていた。製品の本線である「育ったログを開いて話す」経路はどのチェックにも記録されず、`scale` の応答時間も `trace.elapsed_us`（プロセス内の時計。ログ追記・吸収・MKN 同期を含まない）だった。外側の時計 + 応答本文全文を記録するハーネスで測り直した結果:

| ログ | 発話 | vocab | v0.1.7 p50 / p99 | v0.1.8 p50 / p99 |
|---|---|---|---|---|
| fabricate 5k pairs | 10,000 | 103 | 2.93ms / 3.71ms | 1.10ms / 1.34ms |
| fabricate 20k pairs | 40,000 | 103 | 3.92ms / 5.36ms | 1.00ms / 1.39ms |
| fabricate 50k pairs | 100,000 | 103 | 6.70ms / 9.86ms | 0.98ms / 1.19ms |
| fabricate 20k, unique_frac=0.3 | 40,000 | 13,976 | **111.5ms / 445.9ms** | **2.93ms / 4.97ms** |

perf での内訳（vocab 13,976 のシナリオ）: 修正前は自己時間の 59.29% が `NaiveBackoff::distribute`。補間の各次数で語彙サイズ |V| の分布ベクトルを作り直し、その中を線形探索していた。

### 直し方（2 段階）

**段階 1 — 出力バイト同一の書き換え**。240 ターン × 4 シナリオ + probe の応答本文 diff が 0 行であることを確認しながら:

- `next_counts`: SA 範囲内の接尾辞は次トークンで整列済みなので、線形走査 O(occ) を区間二分探索 O(distinct·log occ) に
- sampling unigram（全語彙のソート済み分布）を `Store` にキャッシュ。生成ステップ毎の再構築をコーパス変更時のみに
- smoothing の backoff 参照を線形探索 O(|local|·|V|) から一 pass のハッシュに
- `novelty_lcs`: 全 prior 発話との DP（発話数に比例）を SA 上の最長一致に。SEP/EOS が発話境界を塞ぐため一致が発話をまたげず、値は同一。`prior` フィールド（コーパスの複製）ごと削除
- retrieve: bot 発話の embedding をキャッシュし scan 窓でトリム、MMR の冗長度 max を増分更新、`max_bot_sim` の再 embed も廃止
- open / retokenize: topic 窓（直近 k=5）に入り得る末尾レコードだけ embed

ここまでで小語彙側は発話数に比例した劣化が消えた（1 万→10 万発話で p50 1.10→0.98ms）。大語彙側は p99 47ms で、まだ予算の 30ms を超えていた。

**段階 2 — 疎表現（応答列が変わる。分布は変わらない）**。補間・PPM 除外・skip-gram・recency cache はどれも、触れた id の外では「基底 unigram × スカラー」として作用する。そこで分布を「明示 id の疎な値 + 尾部スカラー」で持ち、|V| ベクトルの実体化を止めた。サンプリングは 2 段（疎部の alias / キャッシュ済み unigram alias + 棄却）。乱数の消費パターンが変わるため**応答列は v0.1.7 と一致しない**。その代わり:

- `sparse_dist_matches_dense_reference`: dense 実装（ツリーに残した参照実装）と全語彙 id の確率を突き合わせ。WB / WB+PPM / KN × 4 文脈で最大誤差 1e-9 + 1e-6·p、L1 < 1e-6
- `sparse_sample_follows_distribution`: 60,000 draw の経験分布が確率 ±5σ 内
- 同一バイナリで 2 回実行した 60 ターン転写が一致（決定性）
- probe / spec 23 / soak 16 / unit 47 すべて PASS

温度 τ≠1・nucleus・top-k を使う場合は dense 経路に落ちる（既定はすべて無効）。

### 恒久化

`munou verify` に `grown-latency-p99` を追加した。`fabricate 4000 pairs, unique_frac=0.3`（release。debug は 500 pairs で、外れたら SKIP）を開き、外側の時計で `respond` を回して p99 ≤ 30ms を判定する。空エンジンの `latency-p99` はこの経路を見ないままなので、両方残している。

## 検証コマンドの読み方

```
PASS / FAIL  … 必須。FAIL があるとプロセスは非ゼロ終了
SKIP         … v0.1 でやらないと決めた項目、または debug では測らない NFR
```

`--sa-tokens` が 10^7 未満だと `rss-1e7` は SKIP。CI は `verify --sa-tokens 20000 --turns 40` と `probe --seed data/seed.jsonl`。verify は `observe-empty` / `observe-logged` / `observe-seed` を含む。
