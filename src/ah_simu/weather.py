"""Deterministic outdoor weather shared by both control policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ah_simu.constants import LATITUDE_DEG, SEASON_DAYS, START_DOY


@dataclass(frozen=True)
class Scenario:
    name: str
    rh_bias: float
    temp_bias: float
    cloud: float
    seed: int


SCENARIOS: dict[str, Scenario] = {
    "baseline": Scenario("baseline", rh_bias=0.0, temp_bias=0.0, cloud=0.28, seed=0),
    "wet": Scenario("wet", rh_bias=16.0, temp_bias=-1.0, cloud=0.55, seed=1),
    "dry": Scenario("dry", rh_bias=-16.0, temp_bias=2.0, cloud=0.12, seed=2),
}


def day_length_hours(doy: int, latitude_deg: float = LATITUDE_DEG) -> float:
    """Day length from Cooper's declination and the sunrise hour angle."""
    decl_deg = 23.45 * np.sin(np.radians(360.0 * (284 + doy) / 365.0))
    lat = np.radians(latitude_deg)
    decl = np.radians(decl_deg)
    cos_hour = -np.tan(lat) * np.tan(decl)
    cos_hour = float(np.clip(cos_hour, -1.0, 1.0))
    return float(24.0 * np.degrees(np.arccos(cos_hour)) / 180.0)


def generate_weather(
    scenario: Scenario,
    days: int = SEASON_DAYS,
    start_doy: int = START_DOY,
    latitude_deg: float = LATITUDE_DEG,
) -> pd.DataFrame:
    rng = np.random.default_rng(scenario.seed)
    rows: list[dict[str, float | int | bool]] = []
    for day_index in range(days):
        doy = ((start_doy - 1 + day_index) % 365) + 1
        day_length = day_length_hours(doy, latitude_deg)
        sunrise = 12.0 - day_length / 2.0
        sunset = 12.0 + day_length / 2.0
        for hour in range(24):
            clock = hour + 0.5
            daylight = sunrise <= clock < sunset
            if daylight:
                phase = (clock - sunrise) / day_length
                radiation = 820.0 * np.sin(np.pi * phase) * (1.0 - scenario.cloud)
            else:
                radiation = 0.0
            diurnal_temp = 7.5 * np.sin(2 * np.pi * (hour - 9) / 24.0)
            diurnal_rh = 16.0 * np.cos(2 * np.pi * (hour - 5) / 24.0)
            outdoor_temp = (
                22.0
                + scenario.temp_bias
                + diurnal_temp
                + float(rng.normal(0.0, 0.35))
            )
            outdoor_rh = (
                68.0
                + scenario.rh_bias
                + diurnal_rh
                - 8.0 * scenario.cloud
                + float(rng.normal(0.0, 1.2))
            )
            rows.append(
                {
                    "day_index": day_index,
                    "hour": hour,
                    "doy": doy,
                    "day_length_h": day_length,
                    "sunrise": sunrise,
                    "sunset": sunset,
                    "is_daylight": daylight,
                    "outdoor_temp_c": outdoor_temp,
                    "outdoor_rh": float(np.clip(outdoor_rh, 25.0, 99.0)),
                    "radiation_wm2": float(max(radiation, 0.0)),
                }
            )
    return pd.DataFrame(rows)
