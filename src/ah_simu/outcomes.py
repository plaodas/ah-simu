"""Cultivation outcomes used to rank the two policies."""

from __future__ import annotations

import pandas as pd

from ah_simu.constants import DISEASE_RH, DISEASE_TEMP_HI, DISEASE_TEMP_LO, SEASON_DAYS


def outcomes(frame: pd.DataFrame) -> dict[str, float]:
    disease = (
        (frame["rh"] >= DISEASE_RH)
        & (frame["temp_c"] >= DISEASE_TEMP_LO)
        & (frame["temp_c"] <= DISEASE_TEMP_HI)
    ).sum()
    flowered = int(frame["flowered"].iloc[-1])
    flowering_day = int(frame["flowering_day"].iloc[-1])
    if flowered and flowering_day >= 0:
        flowering_day = flowering_day + 1
    else:
        flowering_day = int(frame["day_index"].max()) + 1
        flowered = 0
    return {
        "disease_risk_hours": float(disease),
        "irrigation_mm": float(frame["irrigate_mm"].sum()),
        "flowering_day": float(flowering_day),
        "flowered": float(flowered),
        "cover_broken_hours": float(frame["cover_broken"].sum()),
        "inductive_days": float(frame["inductive_days"].iloc[-1]),
        "season_days": float(SEASON_DAYS),
    }
