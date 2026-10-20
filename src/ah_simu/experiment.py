"""Record each scenario and policy as an MLflow run."""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
import pandas as pd

from ah_simu.constants import (
    CRITICAL_LIGHT_HOURS,
    DISEASE_RH,
    DISEASE_TEMP_HI,
    DISEASE_TEMP_LO,
    EXPERIMENT_NAME,
    HORIZON_HOURS,
    HOUSE_NAME,
    LATITUDE_DEG,
    REQUIRED_INDUCTIVE_DAYS,
    SCENARIO_NAMES,
    SEASON_DAYS,
    START_DOY,
    TARGET_LIGHT_HOURS,
)
from ah_simu.house import simulate
from ah_simu.model import ForecastModel, TrainReport
from ah_simu.outcomes import outcomes
from ah_simu.policies import ForecastPolicy, ThresholdPolicy
from ah_simu.weather import SCENARIOS, generate_weather


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def _prepare() -> None:
    mlflow.set_tracking_uri(tracking_uri())
    client = mlflow.MlflowClient()
    if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
        artifact_root = Path(os.environ.get("MLFLOW_ARTIFACT_ROOT", "mlartifacts")).resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)
        client.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_root.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_training(model: ForecastModel, report: TrainReport, model_path: Path) -> str:
    _prepare()
    model.save(str(model_path))
    with mlflow.start_run(run_name="train-forecast") as run:
        mlflow.log_params(
            {
                "house": HOUSE_NAME,
                "model": "ridge-one-step",
                "horizon_hours": HORIZON_HOURS,
                "n_rows": report.n_rows,
            }
        )
        mlflow.log_metrics(
            {
                "rmse_temp": report.rmse_temp,
                "rmse_rh": report.rmse_rh,
                "rmse_soil": report.rmse_soil,
            }
        )
        mlflow.log_artifact(str(model_path), artifact_path="forecast")
        return f"runs:/{run.info.run_id}/forecast/{model_path.name}"


def load_model(path: str | None, model_uri: str | None) -> ForecastModel:
    if path:
        return ForecastModel.load(path)
    if not model_uri:
        raise SystemExit("forecast policy needs --model or --model-uri")
    _prepare()
    local = mlflow.artifacts.download_artifacts(model_uri.strip())
    return ForecastModel.load(local)


def log_evaluation(
    scenario: str,
    policy_name: str,
    frame: pd.DataFrame,
    trajectory: Path,
) -> dict[str, float]:
    _prepare()
    metrics = outcomes(frame)
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(trajectory, index=False)
    spec = SCENARIOS[scenario]
    with mlflow.start_run(run_name=f"{scenario}-{policy_name}"):
        mlflow.log_params(
            {
                "house": HOUSE_NAME,
                "scenario": scenario,
                "policy": policy_name,
                "rh_bias": spec.rh_bias,
                "temp_bias": spec.temp_bias,
                "cloud": spec.cloud,
                "seed": spec.seed,
                "latitude_deg": LATITUDE_DEG,
                "start_doy": START_DOY,
                "season_days": SEASON_DAYS,
                "horizon_hours": HORIZON_HOURS,
                "disease_rh": DISEASE_RH,
                "disease_temp_lo": DISEASE_TEMP_LO,
                "disease_temp_hi": DISEASE_TEMP_HI,
                "target_light_hours": TARGET_LIGHT_HOURS,
                "critical_light_hours": CRITICAL_LIGHT_HOURS,
                "required_inductive_days": REQUIRED_INDUCTIVE_DAYS,
            }
        )
        mlflow.log_metrics(
            {
                "disease_risk_hours": metrics["disease_risk_hours"],
                "irrigation_mm": metrics["irrigation_mm"],
                "flowering_day": metrics["flowering_day"],
                "flowered": metrics["flowered"],
            }
        )
        mlflow.log_artifact(str(trajectory), artifact_path="trajectory")
    return metrics


def run_matrix(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    model, report = ForecastModel.train()
    uri = log_training(model, report, out / "forecast.joblib")
    print(f"trained rmse_rh={report.rmse_rh:.3f} uri={uri}", flush=True)
    summary: dict = {
        "model_uri": uri,
        "rmse_rh": report.rmse_rh,
        "rmse_temp": report.rmse_temp,
        "rmse_soil": report.rmse_soil,
        "scenarios": {},
    }
    for name in SCENARIO_NAMES:
        weather = generate_weather(SCENARIOS[name])
        summary["scenarios"][name] = {}
        for policy in (ThresholdPolicy(), ForecastPolicy(model)):
            frame = simulate(weather, policy)
            frame.insert(0, "scenario", name)
            frame.insert(1, "policy", policy.name)
            trajectory = out / f"{name}_{policy.name}.parquet"
            metrics = log_evaluation(name, policy.name, frame, trajectory)
            summary["scenarios"][name][policy.name] = {
                **metrics,
                "trajectory": trajectory.name,
            }
            print(
                f"{name}/{policy.name} "
                f"disease={metrics['disease_risk_hours']:.0f}h "
                f"water={metrics['irrigation_mm']:.0f}mm "
                f"flower={metrics['flowering_day']:.0f}",
                flush=True,
            )
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out / 'summary.json'}", flush=True)
    return summary
