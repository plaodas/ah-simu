"""Threshold control and a short-horizon forecast control."""

from __future__ import annotations

from ah_simu.constants import (
    FORECAST_IRRIGATE_MM,
    HORIZON_HOURS,
    THRESHOLD_IRRIGATE_MM,
    THRESHOLD_RH,
    THRESHOLD_SOIL,
    THRESHOLD_TEMP_SHADE,
    VENT_BREAKS_COVER,
)
from ah_simu.house import Action, HouseConfig, State
from ah_simu.photoperiod import scheduled_cover


class ThresholdPolicy:
    """React when the current reading crosses a line. Cover is not considered."""

    name = "threshold"
    horizon = 1

    def decide(self, state: State, weather: dict, upcoming: list[dict], config: HouseConfig) -> Action:
        del upcoming, config
        vent = 1.0 if state.rh >= THRESHOLD_RH or state.temp_c >= 32.0 else 0.0
        shade = 0.6 if state.temp_c >= THRESHOLD_TEMP_SHADE else 0.0
        irrigate = THRESHOLD_IRRIGATE_MM if state.soil <= THRESHOLD_SOIL else 0.0
        return Action(vent=vent, irrigate_mm=irrigate, shade=shade, cover=False)


class ForecastPolicy:
    """Pick this hour's actuators from a learned few-hour rollout.

    Later hours in the rollout keep ventilation, irrigation, and shade off, so a
    ventilation choice now is not forced through the covered period.
    """

    name = "forecast"
    horizon = HORIZON_HOURS

    def __init__(self, model) -> None:
        self.model = model

    def decide(self, state: State, weather: dict, upcoming: list[dict], config: HouseConfig) -> Action:
        cover_now = scheduled_cover(
            int(weather["day_index"]),
            float(weather["day_length_h"]),
            int(weather["hour"]),
            bool(weather["is_daylight"]),
        )
        candidates = [
            Action(vent=vent, irrigate_mm=irrigate, shade=shade, cover=cover_now)
            for vent in (0.0, 1.0)
            for irrigate in (0.0, FORECAST_IRRIGATE_MM)
            for shade in (0.0, 0.6)
        ]
        best = min(
            candidates,
            key=lambda action: self._cost(state, weather, upcoming, config, action),
        )
        return best

    def _cost(
        self,
        state: State,
        weather: dict,
        upcoming: list[dict],
        config: HouseConfig,
        action: Action,
    ) -> float:
        from ah_simu.constants import DISEASE_RH, DISEASE_TEMP_HI, DISEASE_TEMP_LO

        predicted = self.model.rollout(state, weather, upcoming, config, action)
        total = 0.15 * action.vent + 0.05 * action.irrigate_mm + 0.02 * action.shade
        if action.cover and action.vent >= VENT_BREAKS_COVER:
            total += 8.0
        for future in predicted:
            if DISEASE_TEMP_LO <= future.temp_c <= DISEASE_TEMP_HI and future.rh >= DISEASE_RH:
                total += 3.0
            total += max(0.0, future.rh - 82.0) * 0.08
            total += max(0.0, 0.38 - future.soil) * 20.0
            total += max(0.0, future.soil - 0.75) * 6.0
        return total
