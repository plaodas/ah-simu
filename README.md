# ダリアハウスの制御比較

外の天候を固定し、ハウスの中の制御だけを変えます。操作はその後の室温・湿度・土壌水分を動かし、終わりに3つの量を並べます。

- **閾値制御**: 今の湿度、または土壌水分が線を超えてから換気・灌水する
- **予測制御**: 湿度と土壌水分の数時間先を見て、被覆を開ける前に動く

見る量は、多湿が続いた時間、灌水量、到花日です。到花は、電照ではなく被覆で日長を目標の 11 時間以内に収めた日が 16 日に達した日です。被覆している日照時間に換気でカーテンを開けると、その日は積算に入りません。

数字はこの比較のための簡略モデルです。実ハウスの収量予測や、灌水・換気・電照の指示には使いません。緯度 36 度、開始日は年間通日 160、56 日です。外気象は「基準」「湿った外気」「乾いた外気」の3つで、乱数の種はシナリオごとに固定しています。

## ローカルで見る

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m ah_simu matrix --out runs
PYTHONPATH=src RUNS_DIR=runs .venv/bin/streamlit run app/streamlit_app.py
```

ブラウザで `http://localhost:8501` を開きます。外気象を切り替え、予測制御と閾値制御で次が違うことを確認します。

- 多湿の時間
- 灌水量
- 到花（湿った外気では、閾値制御は期間内に到花しない）
- 湿度・土壌水分・気温・換気の重なった線

記録は `mlflow.db`（実験名 `dahlia-house-control`）と `mlartifacts/` に残ります。各 run の params にシナリオと方針、metrics に3指標が入ります。

追跡サーバを別に立てる場合は、先に MLflow を起動してから同じコマンドを使います。

```bash
.venv/bin/mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 PYTHONPATH=src \
  .venv/bin/python -m ah_simu matrix --out runs
```

UI は `http://127.0.0.1:5000` です。

## 1台でまとめて起動する

```bash
docker compose up --build
```

MLflow が `http://localhost:5000`、比較画面が `http://localhost:8501` です。初回起動時に、学習と3気象×2方針の評価が終わってから画面が開きます。

## 気象をワークフローで繰り返す

同じ処理は `argo/workflow.yaml` の一本です。

1. `train` が予測モデルを MLflow に置き、artifact URI を次のステップへ渡す
2. `eval` が baseline / wet / dry のそれぞれで閾値制御と予測制御を実行し、3指標を MLflow に残す

```bash
docker build -t ah-simu:local .
argo submit argo/workflow.yaml \
  -p mlflow-uri=http://mlflow.mlflow.svc.cluster.local:5000 \
  -p image=ah-simu:local \
  --watch
```

`mlflow-uri` は、ワークフローの Pod から届く追跡サーバに合わせます。画面用の `runs/summary.json` は、上の `python -m ah_simu matrix` が書きます。

## テスト

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
```
