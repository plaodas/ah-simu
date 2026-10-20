"""One-step ridge model of indoor temperature, humidity, and soil moisture."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ah_simu.constants import EXPLORE_SEED, HORIZON_HOURS, SCENARIO_NAMES
from ah_simu.house import Action, HouseConfig, State, exchange_and_radiation, step
from ah_simu.photoperiod import scheduled_cover
from ah_simu.weather import SCENARIOS, generate_weather

FEATURE_COLUMNS = [
    "temp_c",
    "rh",
    "soil",
    "outdoor_temp_c",
    "outdoor_vp",
    "indoor_vp",
    "radiation",
    "exchange",
    "vp_mixing",
    "temp_mixing",
    "soil_radiation",
    "irrigate_mm",
    "hour_sin",
    "hour_cos",
    "sealed",
]


def _vapor_pressure(temp_c: float, rh: float) -> float:
    svp = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    return float(rh / 100.0 * svp)


def features(state: State, weather: dict, action: Action, config: HouseConfig) -> list[float]:
    return feature_array(state, weather, action, config).tolist()


def feature_array(state: State, weather: dict, action: Action, config: HouseConfig) -> np.ndarray:
    radiation, exchange, sealed = exchange_and_radiation(
        float(weather["radiation_wm2"]), action, config
    )
    indoor_vp = _vapor_pressure(state.temp_c, state.rh)
    outdoor_vp = _vapor_pressure(float(weather["outdoor_temp_c"]), float(weather["outdoor_rh"]))
    outdoor_temp = float(weather["outdoor_temp_c"])
    hour = int(weather["hour"])
    return np.array(
        [
            state.temp_c,
            state.rh,
            state.soil,
            outdoor_temp,
            outdoor_vp,
            indoor_vp,
            radiation,
            exchange,
            (outdoor_vp - indoor_vp) * exchange,
            (outdoor_temp - state.temp_c) * exchange,
            state.soil * radiation,
            action.irrigate_mm,
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            1.0 if sealed else 0.0,
        ],
        dtype=float,
    )


def _explore_action(rng: np.random.Generator, weather: dict) -> Action:
    cover = scheduled_cover(
        int(weather["day_index"]),
        float(weather["day_length_h"]),
        int(weather["hour"]),
        bool(weather["is_daylight"]),
    )
    return Action(
        vent=1.0 if rng.random() < 0.45 else 0.0,
        irrigate_mm=5.0 if rng.random() < 0.08 else 0.0,
        shade=0.6 if rng.random() < 0.12 else 0.0,
        cover=cover,
    )


def collect_transitions(config: HouseConfig | None = None) -> pd.DataFrame:
    config = config or HouseConfig()
    rng = np.random.default_rng(EXPLORE_SEED)
    rows: list[dict] = []
    for name in SCENARIO_NAMES:
        weather = generate_weather(SCENARIOS[name])
        state = State()
        for weather_row in weather.to_dict("records"):
            action = _explore_action(rng, weather_row)
            nxt = step(state, pd.Series(weather_row), action, config)
            row = dict(zip(FEATURE_COLUMNS, features(state, weather_row, action, config), strict=True))
            row["next_temp_c"] = nxt.temp_c
            row["next_rh"] = nxt.rh
            row["next_soil"] = nxt.soil
            rows.append(row)
            state = nxt
    return pd.DataFrame(rows)


def _pipe() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


@dataclass
class TrainReport:
    n_rows: int
    rmse_temp: float
    rmse_rh: float
    rmse_soil: float


def _linear_weights(model: Pipeline) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    scaler = model.named_steps["scale"]
    ridge = model.named_steps["ridge"]
    return (
        np.asarray(scaler.mean_, dtype=float),
        np.asarray(scaler.scale_, dtype=float),
        np.asarray(ridge.coef_, dtype=float),
        float(ridge.intercept_),
    )


class ForecastModel:
    def __init__(self, temp: Pipeline, rh: Pipeline, soil: Pipeline) -> None:
        self.temp = temp
        self.rh = rh
        self.soil = soil
        self._weights = {
            "temp": _linear_weights(temp),
            "rh": _linear_weights(rh),
            "soil": _linear_weights(soil),
        }

    @classmethod
    def train(cls, config: HouseConfig | None = None) -> tuple[ForecastModel, TrainReport]:
        frame = collect_transitions(config)
        rng = np.random.default_rng(0)
        order = rng.permutation(len(frame))
        split = int(len(frame) * 0.8)
        train_idx = order[:split]
        test_idx = order[split:]
        x_train = frame.iloc[train_idx][FEATURE_COLUMNS]
        x_test = frame.iloc[test_idx][FEATURE_COLUMNS]
        models = {}
        rmse = {}
        for key, column in (("temp", "next_temp_c"), ("rh", "next_rh"), ("soil", "next_soil")):
            model = _pipe()
            model.fit(x_train, frame.iloc[train_idx][column])
            pred = model.predict(x_test)
            err = frame.iloc[test_idx][column].to_numpy() - pred
            models[key] = model
            rmse[key] = float(np.sqrt(np.mean(err**2)))
        report = TrainReport(len(frame), rmse["temp"], rmse["rh"], rmse["soil"])
        return cls(models["temp"], models["rh"], models["soil"]), report

    def predict_many(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        def apply(name: str) -> np.ndarray:
            mean, scale, coef, intercept = self._weights[name]
            scaled = (matrix - mean) / scale
            return scaled @ coef + intercept

        temp = np.clip(apply("temp"), 5.0, 45.0)
        rh = np.clip(apply("rh"), 5.0, 100.0)
        soil = np.clip(apply("soil"), 0.05, 0.95)
        return temp, rh, soil

    def predict_next(self, state: State, weather: dict, action: Action, config: HouseConfig) -> State:
        matrix = feature_array(state, weather, action, config).reshape(1, -1)
        temp, rh, soil = self.predict_many(matrix)
        return State(
            temp_c=float(temp[0]),
            rh=float(rh[0]),
            soil=float(soil[0]),
            inductive_days=state.inductive_days,
            flowered=state.flowered,
            flowering_day=state.flowering_day,
        )

    def rollout(
        self,
        state: State,
        weather: dict,
        upcoming: list[dict],
        config: HouseConfig,
        action: Action,
    ) -> list[State]:
        predicted = [self.predict_next(state, weather, action, config)]
        cursor = predicted[0]
        for future in upcoming[: HORIZON_HOURS - 1]:
            cover = scheduled_cover(
                int(future["day_index"]),
                float(future["day_length_h"]),
                int(future["hour"]),
                bool(future["is_daylight"]),
            )
            held = Action(vent=0.0, irrigate_mm=0.0, shade=0.0, cover=cover)
            cursor = self.predict_next(cursor, future, held, config)
            predicted.append(cursor)
        return predicted

    def save(self, path: str) -> None:
        dump({"temp": self.temp, "rh": self.rh, "soil": self.soil}, path)

    @classmethod
    def load(cls, path: str) -> ForecastModel:
        payload = load(path)
        return cls(payload["temp"], payload["rh"], payload["soil"])
