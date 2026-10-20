"""Shared assumptions for the simplified dahlia house.

The plant response ranks control policies inside this simulation.
It is not a field recommendation.
"""

LATITUDE_DEG = 36.0
START_DOY = 160
SEASON_DAYS = 56
VEG_DAYS = 10
TARGET_LIGHT_HOURS = 11.0
CRITICAL_LIGHT_HOURS = 11.5
REQUIRED_INDUCTIVE_DAYS = 16
HORIZON_HOURS = 6

# Gray-mold-like hours: cool-to-mild and humid. A ranking device, not a diagnosis.
DISEASE_RH = 88.0
DISEASE_TEMP_LO = 12.0
DISEASE_TEMP_HI = 24.0

THRESHOLD_RH = 85.0
THRESHOLD_SOIL = 0.34
THRESHOLD_TEMP_SHADE = 30.0
THRESHOLD_IRRIGATE_MM = 6.0

FORECAST_IRRIGATE_MM = 4.0
VENT_BREAKS_COVER = 0.5

EXPLORE_SEED = 7
HOUSE_NAME = "house-1"
EXPERIMENT_NAME = "dahlia-house-control"
SCENARIO_NAMES = ("baseline", "wet", "dry")
