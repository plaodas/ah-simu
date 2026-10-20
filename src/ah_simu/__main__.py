"""CLI for the train → evaluate → record sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

from ah_simu.constants import SCENARIO_NAMES
from ah_simu.experiment import load_model, log_evaluation, log_training, run_matrix
from ah_simu.house import simulate
from ah_simu.model import ForecastModel
from ah_simu.policies import ForecastPolicy, ThresholdPolicy
from ah_simu.weather import SCENARIOS, generate_weather


def _train(args: argparse.Namespace) -> None:
    model, report = ForecastModel.train()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    uri = log_training(model, report, output)
    print(
        f"trained rows={report.n_rows} rmse_rh={report.rmse_rh:.3f} uri={uri}",
        flush=True,
    )
    if args.uri_file:
        Path(args.uri_file).write_text(uri + "\n", encoding="utf-8")


def _eval(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    weather = generate_weather(SCENARIOS[args.scenario])
    if args.policy == "threshold":
        policy = ThresholdPolicy()
    else:
        policy = ForecastPolicy(load_model(args.model, args.model_uri))
    frame = simulate(weather, policy)
    frame.insert(0, "scenario", args.scenario)
    frame.insert(1, "policy", policy.name)
    trajectory = out / f"{args.scenario}_{policy.name}.parquet"
    metrics = log_evaluation(args.scenario, policy.name, frame, trajectory)
    print(
        f"{args.scenario}/{policy.name} "
        f"disease={metrics['disease_risk_hours']:.0f}h "
        f"water={metrics['irrigation_mm']:.0f}mm "
        f"flower={metrics['flowering_day']:.0f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dahlia house control policies")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train", help="Fit the humidity and soil forecast and log it")
    train.add_argument("--output", default="forecast.joblib")
    train.add_argument("--uri-file", help="Write the MLflow model URI for the next step")
    train.set_defaults(func=_train)

    evaluate = sub.add_parser("eval", help="Run one scenario and policy, then log metrics")
    evaluate.add_argument("--scenario", required=True, choices=SCENARIO_NAMES)
    evaluate.add_argument("--policy", required=True, choices=("threshold", "forecast"))
    evaluate.add_argument("--model", help="Local forecast.joblib")
    evaluate.add_argument("--model-uri", help="MLflow artifact URI from the train step")
    evaluate.add_argument("--out", default="runs")
    evaluate.set_defaults(func=_eval)

    matrix = sub.add_parser("matrix", help="Train once, then evaluate every weather pattern")
    matrix.add_argument("--out", default="runs")
    matrix.set_defaults(func=lambda args: run_matrix(Path(args.out)))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
