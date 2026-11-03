"""A 2024 forecast of X/Y speed can only use the 2023 profile."""

import pandas as pd

from experiments.a1_multistint_midfield import FRONT_PART, PART
from experiments.a1_profile_future_pace import prior_mean_speed


def target(source):
    return prior_mean_speed(source).query("season == 2024").sort_values(["circuit", "driver"]).reset_index(drop=True)


def test_synthetic_cutoff():
    source = pd.DataFrame([
        ("Bahrain", 2023, 1, 0, 0, 200.0), ("Bahrain", 2023, 2, 0, 0, 190.0),
        ("Bahrain", 2024, 1, 0, 0, 210.0), ("Bahrain", 2024, 2, 0, 0, 195.0),
    ], columns=["circuit", "season", "driver", "half", "bin", "speed"])
    expected = target(source)
    changed = source.copy(); changed.loc[changed.season >= 2024, "speed"] = 999
    pd.testing.assert_frame_equal(expected, target(changed), check_exact=True)
    changed.loc[(changed.season == 2023) & (changed.driver == 1), "speed"] = 220
    assert not expected.prior_xy_speed.equals(target(changed).prior_xy_speed)


def test_cached_target_scramble():
    source = pd.concat([pd.read_csv(FRONT_PART), pd.read_csv(PART)], ignore_index=True)
    expected = target(source)
    changed = source.copy(); changed.loc[changed.season >= 2024, "speed"] = 999
    pd.testing.assert_frame_equal(expected, target(changed), check_exact=True)


if __name__ == "__main__":
    test_synthetic_cutoff()
    test_cached_target_scramble()
    print("profile future-pace leak test passed")
