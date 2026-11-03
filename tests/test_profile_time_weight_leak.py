"""A target-year time-weighted X/Y feature uses only the earlier profile."""

import pandas as pd

from experiments.a1_multistint_midfield import FRONT_PART, PART
from experiments.a1_profile_time_weight import prior_time_advantage


def target(source):
    return prior_time_advantage(source).query("season == 2024").sort_values(["circuit", "driver"]).reset_index(drop=True)


def test_synthetic_cutoff():
    source = pd.DataFrame([
        ("Bahrain", 2023, 1, 0, 0, 200.0), ("Bahrain", 2023, 2, 0, 0, 190.0),
        ("Bahrain", 2024, 1, 0, 0, 210.0), ("Bahrain", 2024, 2, 0, 0, 195.0),
    ], columns=["circuit", "season", "driver", "half", "bin", "speed"])
    expected = target(source)
    changed = source.copy(); changed.loc[changed.season >= 2024, "speed"] = 999
    pd.testing.assert_frame_equal(expected, target(changed), check_exact=True)
    changed.loc[(changed.season == 2023) & (changed.driver == 1), "speed"] = 220
    assert not expected.prior_xy_time_adv.equals(target(changed).prior_xy_time_adv)


def test_cached_target_scramble():
    source = pd.concat([pd.read_csv(FRONT_PART), pd.read_csv(PART)], ignore_index=True)
    expected = target(source)
    changed = source.copy(); changed.loc[changed.season >= 2024, "speed"] = 999
    pd.testing.assert_frame_equal(expected, target(changed), check_exact=True)


if __name__ == "__main__":
    test_synthetic_cutoff()
    test_cached_target_scramble()
    print("profile time-weight leak test passed")
