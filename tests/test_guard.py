"""The harness must refuse pre-weekend runs that are handed target-weekend or label columns (this exact bug produced a bogus 0.71 in E30/E31)."""
import f1.walkforward as W
import pandas as pd


WEATHER_BAD = ("rain_act", "temp_act", "wind_act", "fc3_rain", "fc3_temp",
               "wetrace", "rain_x_skill", "rain_x_form", "rain_x_gridless")


def test_guard_blocks_target_columns():
    for bad in ("y_pace", "gridn", "q_gap", "podium", "fp_lr", "cur_apex_slow", "led_share") + WEATHER_BAD:
        try:
            W.assert_pre_weekend(["d3_order", bad])
        except ValueError:
            continue
        raise AssertionError(f"guard did not block {bad}")
    W.assert_pre_weekend(["d3_order", "t10_pace_gap", "tp_apex_slow", "fit_car_apex", "c_dnf", "d_wet_skill", "d3_temp_act"])


def test_harness_rejects_weather_before_accessing_data():
    for bad in WEATHER_BAD:
        try:
            W.walk_forward(pd.DataFrame(), [2020], ["missing_model"], F=[bad], tie=None)
        except ValueError as exc:
            assert bad in str(exc)
            continue
        raise AssertionError(f"harness did not block {bad}")


if __name__ == "__main__":
    test_guard_blocks_target_columns()
    test_harness_rejects_weather_before_accessing_data()
    print("guard ok")
