"""The harness must refuse pre-weekend runs that are handed target-weekend or label columns (this exact bug produced a bogus 0.71 in E30/E31)."""
import pytest
import f1.walkforward as W


def test_guard_blocks_target_columns():
    for bad in ("y_pace", "gridn", "q_gap", "podium", "fp_lr", "cur_apex_slow", "led_share"):
        with pytest.raises(ValueError):
            W.assert_pre_weekend(["d3_order", bad])
    W.assert_pre_weekend(["d3_order", "t10_pace_gap", "tp_apex_slow", "fit_car_apex", "c_dnf"])


if __name__ == "__main__":
    test_guard_blocks_target_columns(); print("guard ok")
