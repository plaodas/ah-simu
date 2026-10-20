"""One greenhouse. Actuators change the next indoor state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ah_simu.photoperiod import day_is_inductive, hour_is_light, scheduled_cover


def saturation_vapor_pressure(temp_c: float) -> float:
    return float(6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5)))


@dataclass(frozen=True)
class HouseConfig:
    conduction: float = 0.18
    leakiness: float = 0.08
    leak_sealed: float = 0.015
    vent_exchange: float = 0.7
    solar_heat: float = 0.0045
    et_sun: float = 0.018
    et_base: float = 0.0035
    et_to_vp: float = 55.0
    mm_to_soil: float = 0.016
    field_capacity: float = 0.68
    drainage: float = 0.4


@dataclass
class State:
    temp_c: float = 22.0
    rh: float = 72.0
    soil: float = 0.48
    inductive_days: int = 0
    flowered: bool = False
    flowering_day: int | None = None


@dataclass(frozen=True)
class Action:
    vent: float
    irrigate_mm: float
    shade: float
    cover: bool


def exchange_and_radiation(
    radiation_wm2: float,
    action: Action,
    config: HouseConfig,
) -> tuple[float, float, bool]:
    """Return effective radiation, air-exchange rate, and whether the cover stayed sealed."""
    sealed = action.cover and action.vent < 0.5
    if sealed:
        return 0.0, config.leak_sealed, True
    radiation = radiation_wm2 * (1.0 - action.shade)
    exchange = config.leakiness + action.vent * config.vent_exchange
    return radiation, exchange, False


def step(state: State, weather: pd.Series, action: Action, config: HouseConfig) -> State:
    radiation, exchange, _sealed = exchange_and_radiation(
        float(weather["radiation_wm2"]), action, config
    )
    outdoor_temp = float(weather["outdoor_temp_c"])
    temp = state.temp_c + (config.conduction + exchange) * (outdoor_temp - state.temp_c)
    temp += config.solar_heat * radiation
    temp = float(np.clip(temp, 5.0, 45.0))

    vp = (state.rh / 100.0) * saturation_vapor_pressure(state.temp_c)
    outdoor_vp = (float(weather["outdoor_rh"]) / 100.0) * saturation_vapor_pressure(outdoor_temp)
    vp = vp + exchange * (outdoor_vp - vp)
    light_factor = radiation / 800.0
    et = (config.et_sun * light_factor + config.et_base) * state.soil
    vp = vp + config.et_to_vp * et
    rh = float(np.clip(100.0 * vp / saturation_vapor_pressure(temp), 5.0, 100.0))

    soil = state.soil - et + action.irrigate_mm * config.mm_to_soil
    if soil > config.field_capacity:
        soil -= config.drainage * (soil - config.field_capacity)
    soil = float(np.clip(soil, 0.05, 0.95))

    return State(
        temp_c=temp,
        rh=rh,
        soil=soil,
        inductive_days=state.inductive_days,
        flowered=state.flowered,
        flowering_day=state.flowering_day,
    )


def simulate(
    weather: pd.DataFrame,
    policy,
    config: HouseConfig | None = None,
) -> pd.DataFrame:
    """Run one policy on a fixed weather frame. `policy.decide` sees state and future hours."""
    config = config or HouseConfig()
    state = State()
    rows: list[dict] = []
    light_hours = 0.0
    current_day = -1

    records = weather.to_dict("records")
    for index, weather_row in enumerate(records):
        day_index = int(weather_row["day_index"])
        if day_index != current_day:
            if current_day >= 0:
                state = _close_day(state, current_day, light_hours)
            current_day = day_index
            light_hours = 0.0

        upcoming = records[index + 1 : index + 1 + policy.horizon]
        action = policy.decide(state, weather_row, upcoming, config)
        cover = scheduled_cover(
            day_index,
            float(weather_row["day_length_h"]),
            int(weather_row["hour"]),
            bool(weather_row["is_daylight"]),
        )
        action = Action(action.vent, action.irrigate_mm, action.shade, cover)
        if hour_is_light(bool(weather_row["is_daylight"]), cover, action.vent):
            light_hours += 1.0

        nxt = step(state, pd.Series(weather_row), action, config)
        rows.append(
            {
                "day_index": day_index,
                "hour": int(weather_row["hour"]),
                "doy": int(weather_row["doy"]),
                "outdoor_temp_c": float(weather_row["outdoor_temp_c"]),
                "outdoor_rh": float(weather_row["outdoor_rh"]),
                "radiation_wm2": float(weather_row["radiation_wm2"]),
                "temp_c": nxt.temp_c,
                "rh": nxt.rh,
                "soil": nxt.soil,
                "vent": action.vent,
                "irrigate_mm": action.irrigate_mm,
                "shade": action.shade,
                "cover": int(cover),
                "cover_broken": int(cover and action.vent >= 0.5),
                "inductive_days": nxt.inductive_days,
                "flowering_day": -1 if nxt.flowering_day is None else nxt.flowering_day,
            }
        )
        state = nxt

    if current_day >= 0:
        state = _close_day(state, current_day, light_hours)
        if rows:
            rows[-1]["inductive_days"] = state.inductive_days
            rows[-1]["flowering_day"] = -1 if state.flowering_day is None else state.flowering_day

    frame = pd.DataFrame(rows)
    flowering = state.flowering_day
    frame["flowering_day"] = -1 if flowering is None else flowering
    frame["flowered"] = int(flowering is not None)
    frame["inductive_days"] = state.inductive_days
    return frame


def _close_day(state: State, day_index: int, light_hours: float) -> State:
    inductive = state.inductive_days + (1 if day_is_inductive(light_hours) else 0)
    flowered = state.flowered
    flowering_day = state.flowering_day
    if not flowered and inductive >= state_required_days():
        flowered = True
        flowering_day = day_index
    return State(
        temp_c=state.temp_c,
        rh=state.rh,
        soil=state.soil,
        inductive_days=inductive,
        flowered=flowered,
        flowering_day=flowering_day,
    )


def state_required_days() -> int:
    from ah_simu.constants import REQUIRED_INDUCTIVE_DAYS

    return REQUIRED_INDUCTIVE_DAYS
