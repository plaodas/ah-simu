"""Overlay the two control policies for one shared weather pattern."""

from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

SCENARIO_LABELS = {
    "baseline": "基準",
    "wet": "湿った外気",
    "dry": "乾いた外気",
}
POLICY_LABELS = {"threshold": "閾値制御", "forecast": "予測制御"}
POLICY_COLORS = {"閾値制御": "#c2410c", "予測制御": "#0f766e"}


def runs_dir() -> Path:
    return Path(os.environ.get("RUNS_DIR", "runs"))


def load_summary(directory: Path) -> dict:
    return json.loads((directory / "summary.json").read_text(encoding="utf-8"))


def load_pair(directory: Path, scenario: str, summary: dict) -> pd.DataFrame:
    frames = []
    for policy in ("threshold", "forecast"):
        name = summary["scenarios"][scenario][policy]["trajectory"]
        frame = pd.read_parquet(directory / name)
        frame["方針"] = POLICY_LABELS[policy]
        frame["栽培日"] = frame["day_index"] + frame["hour"] / 24.0
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def flower_label(metrics: dict) -> str:
    if metrics["flowered"] < 1:
        return "期間内に到花せず"
    return f"{int(metrics['flowering_day'])}日目"


def chart(frame: pd.DataFrame, y: str, title: str) -> alt.Chart:
    return (
        alt.Chart(frame)
        .mark_line()
        .encode(
            x=alt.X("栽培日:Q", title="栽培日"),
            y=alt.Y(f"{y}:Q", title=title),
            color=alt.Color(
                "方針:N",
                title="方針",
                scale=alt.Scale(
                    domain=list(POLICY_COLORS),
                    range=list(POLICY_COLORS.values()),
                ),
            ),
        )
        .properties(height=220)
    )


def main() -> None:
    st.set_page_config(page_title="ダリアハウス 制御比較", layout="wide")
    st.title("ダリアハウス — 同じ天候で制御を変える")
    st.caption(
        "外の天候は共通です。閾値制御は今の湿度・土壌水分が線を超えてから動きます。"
        "予測制御は数時間先の湿度と土壌水分を見て、被覆を開ける前に換気や灌水をします。"
        "植物の反応はこの比較のための簡略モデルで、実ハウスの灌水・換気・電照の指示には使いません。"
    )

    directory = runs_dir()
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        st.warning("比較結果がまだありません。次を実行してから、この画面を開き直してください。")
        st.code("PYTHONPATH=src python -m ah_simu matrix --out runs", language="bash")
        return

    summary = load_summary(directory)
    scenario = st.selectbox(
        "外気象",
        options=list(SCENARIO_LABELS),
        format_func=lambda key: SCENARIO_LABELS[key],
    )
    pair = summary["scenarios"][scenario]
    threshold = pair["threshold"]
    forecast = pair["forecast"]

    disease_delta = forecast["disease_risk_hours"] - threshold["disease_risk_hours"]
    water_delta = forecast["irrigation_mm"] - threshold["irrigation_mm"]
    flower_delta = forecast["flowering_day"] - threshold["flowering_day"]

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "多湿の時間",
        f"{forecast['disease_risk_hours']:.0f} h",
        delta=f"{disease_delta:.0f} h 対 閾値 {threshold['disease_risk_hours']:.0f} h",
        delta_color="inverse",
    )
    c2.metric(
        "灌水量",
        f"{forecast['irrigation_mm']:.0f} mm",
        delta=f"{water_delta:.0f} mm 対 閾値 {threshold['irrigation_mm']:.0f} mm",
        delta_color="inverse",
    )
    c3.metric(
        "到花",
        flower_label(forecast),
        delta=f"{flower_delta:.0f} 日 対 閾値 {flower_label(threshold)}",
        delta_color="inverse",
    )
    st.caption(
        "表示の主値は予測制御です。差は予測制御から閾値制御を引いた値で、"
        "負の値は予測制御のほうが小さいことを示します。"
        f" 1時間先湿度の学習誤差（RMSE）は {summary['rmse_rh']:.2f} ポイントです。"
        "到花は、被覆で日長を目標以内に収めた日が規定数に達した栽培日です。"
        "被覆中に換気でカーテンを開けるとその日は到花の積算に入りません。"
    )

    frame = load_pair(directory, scenario, summary)
    left, right = st.columns(2)
    with left:
        st.altair_chart(chart(frame, "rh", "室内相対湿度 (%)"), width="stretch")
        st.altair_chart(chart(frame, "temp_c", "室内気温 (℃)"), width="stretch")
    with right:
        st.altair_chart(chart(frame, "soil", "土壌水分 (0–1)"), width="stretch")
        st.altair_chart(chart(frame, "vent", "換気 (0 閉 / 1 開)"), width="stretch")

    frame["灌水累計"] = frame.groupby("方針", sort=False)["irrigate_mm"].cumsum()
    st.altair_chart(chart(frame, "灌水累計", "灌水の累計 (mm)"), width="stretch")


main()
