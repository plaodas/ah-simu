"""Short-day schedule for cut-flower dahlia in this simulation.

Summer days are longer than the target light window. The curtain covers
daylight outside that window. Opening the curtain (ventilation at or above
the break level) during those covered daylight hours counts as light.
"""

from __future__ import annotations

from ah_simu.constants import (
    CRITICAL_LIGHT_HOURS,
    TARGET_LIGHT_HOURS,
    VENT_BREAKS_COVER,
    VEG_DAYS,
)


def light_window(day_length_h: float) -> tuple[float, float]:
    """Intended uncovered hours, narrowed when the sun is already short."""
    target = min(TARGET_LIGHT_HOURS, day_length_h)
    return 12.0 - target / 2.0, 12.0 + target / 2.0


def scheduled_cover(day_index: int, day_length_h: float, hour: int, is_daylight: bool) -> bool:
    if day_index < VEG_DAYS or not is_daylight:
        return False
    start, end = light_window(day_length_h)
    clock = hour + 0.5
    return clock < start or clock >= end


def hour_is_light(is_daylight: bool, cover: bool, vent: float) -> bool:
    if not is_daylight:
        return False
    if cover and vent < VENT_BREAKS_COVER:
        return False
    return True


def day_is_inductive(light_hours: float) -> bool:
    return light_hours <= CRITICAL_LIGHT_HOURS
