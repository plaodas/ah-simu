import pandas as pd

from ah_simu.house import Action, HouseConfig, State, simulate, step
from ah_simu.model import ForecastModel
from ah_simu.outcomes import outcomes
from ah_simu.photoperiod import day_is_inductive, hour_is_light, scheduled_cover
from ah_simu.policies import ForecastPolicy, ThresholdPolicy
from ah_simu.weather import SCENARIOS, day_length_hours, generate_weather


def _row(**overrides) -> pd.Series:
    base = {
        "radiation_wm2": 0.0,
        "outdoor_temp_c": 20.0,
        "outdoor_rh": 50.0,
    }
    base.update(overrides)
    return pd.Series(base)


def test_day_length_solstice():
    assert 14.2 < day_length_hours(172, 36.0) < 14.8
    assert 9.4 < day_length_hours(355, 36.0) < 10.1


def test_cover_keeps_a_short_day_unless_ventilation_opens_it():
    assert scheduled_cover(0, 14.5, 5, True) is False
    assert scheduled_cover(20, 14.5, 5, True) is True
    assert scheduled_cover(20, 14.5, 12, True) is False
    assert hour_is_light(True, True, 0.0) is False
    assert hour_is_light(True, True, 1.0) is True
    assert day_is_inductive(11.0)
    assert day_is_inductive(12.0) is False


def test_shade_cuts_the_temperature_rise():
    config = HouseConfig()
    state = State(temp_c=22.0, rh=50.0, soil=0.40)
    row = _row(radiation_wm2=800.0, outdoor_temp_c=24.0, outdoor_rh=40.0)
    full_sun = step(state, row, Action(0.0, 0.0, 0.0, False), config)
    shaded = step(state, row, Action(0.0, 0.0, 0.6, False), config)
    assert shaded.temp_c < full_sun.temp_c


def test_irrigation_raises_soil_and_ventilation_lowers_humidity():
    config = HouseConfig()
    state = State(temp_c=22.0, rh=90.0, soil=0.30)
    dry = step(state, _row(), Action(0.0, 0.0, 0.0, False), config)
    watered = step(state, _row(), Action(0.0, 10.0, 0.0, False), config)
    opened = step(state, _row(outdoor_rh=40.0), Action(1.0, 0.0, 0.0, False), config)
    assert watered.soil > dry.soil
    assert opened.rh < dry.rh


def test_same_weather_changes_indoor_state_when_the_policy_changes():
    weather = generate_weather(SCENARIOS["wet"])
    model, _report = ForecastModel.train()
    threshold = simulate(weather, ThresholdPolicy())
    again = simulate(weather, ThresholdPolicy())
    forecast = simulate(weather, ForecastPolicy(model))
    pd.testing.assert_series_equal(threshold["outdoor_temp_c"], again["outdoor_temp_c"])
    pd.testing.assert_frame_equal(threshold, again)
    pd.testing.assert_series_equal(threshold["outdoor_rh"], forecast["outdoor_rh"])
    assert not threshold["rh"].equals(forecast["rh"])

    wet_threshold = outcomes(threshold)
    wet_forecast = outcomes(forecast)
    assert wet_forecast["disease_risk_hours"] < wet_threshold["disease_risk_hours"]
    assert wet_forecast["flowering_day"] < wet_threshold["flowering_day"]
    assert wet_forecast["flowered"] == 1
    assert wet_forecast["irrigation_mm"] != wet_threshold["irrigation_mm"]
