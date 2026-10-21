# AGENTS.md

## このリポジトリの目的

同じ合成外気象の下で、ダリア栽培ハウスの閾値制御と予測制御を比較する Python シミュレータです。
評価指標は多湿時間、灌水量、到花日です。これは方針比較用の簡略モデルであり、実際の栽培指示・病害診断・収量予測には使いません。

## 構成

- `src/ah_simu/weather.py`: 固定 seed の合成外気象
- `src/ah_simu/house.py`: 1 時間刻みの状態遷移と栽培期間のシミュレーション
- `src/ah_simu/photoperiod.py`: 被覆、実効日長、短日成立の判定
- `src/ah_simu/policies.py`: 閾値制御と Ridge 予測を使う制御
- `src/ah_simu/model.py`: 学習データ生成、1 時間先モデル、複数時間 rollout
- `src/ah_simu/outcomes.py`: 比較指標の集計
- `src/ah_simu/experiment.py`: 全条件の実行、Parquet/JSON 出力、MLflow 記録
- `src/ah_simu/__main__.py`: `train`、`eval`、`matrix` CLI
- `app/streamlit_app.py`: `runs/` の結果を表示する Streamlit UI
- `tests/`: シミュレーションの性質と実験パイプラインのテスト
- `argo/workflow.yaml`: 学習後に 3 気象 × 2 方針を評価する Argo Workflow

## セットアップと検証

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
```

テスト設定に `pythonpath = src` があるため、pytest には通常 `PYTHONPATH` は不要です。CLI と UI は次のように実行します。

```bash
PYTHONPATH=src .venv/bin/python -m ah_simu matrix --out runs
PYTHONPATH=src RUNS_DIR=runs .venv/bin/streamlit run app/streamlit_app.py
```

変更後は少なくとも `.venv/bin/pytest` を実行してください。実験結果、指標、出力スキーマ、CLI、MLflow 記録を変えた場合は `matrix` も実行し、`runs/summary.json` と 6 個の軌跡が生成されることを確認します。UI を変えた場合は Streamlit 画面も確認します。

## 守るべきモデル上の性質

- 両方針には同じ気象 DataFrame を渡し、公平な比較を維持する。
- 気象と学習データの固定 seed を不用意に変えず、テストとデモを再現可能に保つ。
- 状態遷移は 1 時間刻みである。単位（℃、相対湿度 %、mm、0–1 の土壌水分、時間）を暗黙に変更しない。
- 被覆は `scheduled_cover` が決め、方針が返した `cover` は `simulate` で上書きされる。被覆中の換気は光として数え、短日成立と到花日に影響する。
- 到花日は 0-based の内部日番号ではなく、`outcomes` で 1-based の表示値に変換する。期間内に到花しない場合は `season_days` を返し、`flowered` で区別する。
- 予測制御は 8 操作候補を比較し、最初の 1 時間だけ候補操作を適用する。rollout の後続時間は換気・灌水・遮光を停止する。
- 指標や定数を変えるときは、実装、テスト、README の説明、Streamlit の文言を一緒に更新する。

## 実装方針

- 既存の型ヒント、dataclass、小さな純粋関数、モジュール docstring のスタイルに合わせる。
- 共有する数値は `constants.py` に置き、同じ閾値を複数モジュールへ直書きしない。
- DataFrame の列名は CLI 出力、Parquet、UI、テスト間の契約として扱う。変更時は全利用箇所を追跡する。
- 新しい気象シナリオは `SCENARIOS` と `SCENARIO_NAMES` を整合させ、UI の日本語ラベル、Argo の評価行列、テストも更新する。
- 新しい方針は `name`、`horizon`、`decide(state, weather, upcoming, config)` を提供し、同一天候で比較するテストを追加する。
- 実際の栽培に有効だと断定しない。README と UI では簡略モデルである旨を維持する。
- フォーマッタやリンタは現在設定されていない。無関係な一括整形や新規ツール導入は避ける。

## 生成物と外部サービス

- `runs/`、`mlflow.db`、`mlartifacts/`、`*.joblib` は生成物であり、コミットしない。
- 通常のローカル実行は SQLite の `mlflow.db` と `mlartifacts/` を使う。外部 MLflow は `MLFLOW_TRACKING_URI` で指定する。
- `compose.yaml` は MLflow と Streamlit を起動し、結果がなければ `matrix` を一度実行する。
- Argo の既定 MLflow URI はクラスタ内 DNS であり、ローカル WSL から利用できる前提にしない。

## 変更時のテスト観点

- 物理モデル: 換気、遮光、灌水が期待する向きに状態を変えること
- 日長モデル: 被覆中の換気が短日成立を壊すこと
- 比較: 同じ方針・seed は同じ結果になり、両方針の外気象列は一致すること
- 実験: 3 シナリオ × 2 方針の結果と MLflow run が作られること
- ワークフロー: 学習 run の model URI が各 forecast 評価へ渡ること
