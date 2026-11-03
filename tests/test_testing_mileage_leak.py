"""Pre-season mileage uses only testing laps and the known entry list."""

import pandas as pd

from experiments.a6_testing_mileage import team_mileage


def test_testing_mileage_ignores_target_and_later():
    entries = pd.DataFrame([
        (2023, 1, "A", "red", 1), (2023, 1, "B", "red", 2),
        (2023, 1, "C", "blue", 3), (2024, 1, "A", "red", 4),
    ], columns=["season", "round", "code", "team", "order"])
    source = pd.DataFrame([
        (2023, 1, "A", 90), (2023, 1, "B", 80), (2023, 1, "C", 70),
        (2024, 1, "A", 100),
    ], columns=["season", "day", "code", "laps"])
    target = lambda d: d[d.season == 2023].sort_values("team").reset_index(drop=True)
    expected = target(team_mileage(entries, source))
    changed_entries = entries.copy()
    changed_entries.loc[changed_entries.season >= 2023, "order"] = 99
    changed_source = source.copy()
    changed_source.loc[changed_source.season >= 2024, "laps"] = 999
    pd.testing.assert_frame_equal(expected, target(team_mileage(changed_entries, changed_source)))
    changed_source.loc[(changed_source.season == 2023) & (changed_source.code == "A"), "laps"] = 95
    assert target(team_mileage(entries, changed_source)).loc[1, "laps"] == 175


if __name__ == "__main__":
    test_testing_mileage_ignores_target_and_later()
    print("testing mileage leak test passed")
