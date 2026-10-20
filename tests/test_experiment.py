from pathlib import Path

import mlflow

from ah_simu.constants import EXPERIMENT_NAME, SCENARIO_NAMES
from ah_simu.experiment import load_model, run_matrix
from ah_simu.house import Action, HouseConfig, State


def test_matrix_logs_three_metrics_for_each_weather(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(tmp_path / "mlartifacts"))
    out = tmp_path / "runs"
    summary = run_matrix(out)

    assert set(summary["scenarios"]) == set(SCENARIO_NAMES)
    for scenario in SCENARIO_NAMES:
        for policy in ("threshold", "forecast"):
            metrics = summary["scenarios"][scenario][policy]
            assert metrics["disease_risk_hours"] >= 0
            assert metrics["irrigation_mm"] > 0
            assert metrics["flowering_day"] >= 1
            assert (out / metrics["trajectory"]).exists()

    wet = summary["scenarios"]["wet"]
    assert wet["forecast"]["disease_risk_hours"] < wet["threshold"]["disease_risk_hours"]
    assert wet["forecast"]["flowering_day"] < wet["threshold"]["flowering_day"]

    loaded = load_model(None, summary["model_uri"])
    predicted = loaded.predict_next(
        State(),
        {
            "radiation_wm2": 0.0,
            "outdoor_temp_c": 20.0,
            "outdoor_rh": 60.0,
            "hour": 4,
        },
        Action(0.0, 0.0, 0.0, False),
        HouseConfig(),
    )
    assert 5.0 <= predicted.rh <= 100.0

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    frame = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(frame) == 1 + len(SCENARIO_NAMES) * 2
    for column in ("metrics.disease_risk_hours", "metrics.irrigation_mm", "metrics.flowering_day"):
        assert column in frame.columns


def test_workflow_passes_the_model_uri_into_each_weather():
    text = (Path(__file__).resolve().parents[1] / "argo" / "workflow.yaml").read_text(encoding="utf-8")
    assert "steps.train.outputs.parameters.model-uri" in text
    for scenario in SCENARIO_NAMES:
        assert scenario in text
    assert "threshold" in text and "forecast" in text
    assert "--uri-file" in text
    assert "--model-uri" in text
