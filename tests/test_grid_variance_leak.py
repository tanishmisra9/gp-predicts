"""Grid-uncertainty features must ignore the target race's actual grid."""

import pandas as pd

from experiments.b8_grid_variance import prior_variance


def test_target_grid_scramble():
    rows = []
    for season, k in ((2019, 1), (2020, 2)):
        for i in range(6):
            rows.append(dict(season=season, round=1, k=k, driver=f"driver{i}",
                             code=f"D{i}", d_season_pts=6 - i, d10_order=i + 1,
                             prior_grid=i + 1.5, gridn=i + 1))
    source = pd.DataFrame(rows)
    expected = prior_variance(source).sort_values("driver").reset_index(drop=True)
    changed = source.copy()
    changed.loc[changed.season == 2020, "gridn"] = 21
    actual = prior_variance(changed).sort_values("driver").reset_index(drop=True)
    pd.testing.assert_frame_equal(expected[["global_var", "driver_var"]],
                                  actual[["global_var", "driver_var"]])
    changed.loc[changed.season == 2019, "gridn"] = [21] * 6
    earlier_edit = prior_variance(changed)
    assert not earlier_edit.global_var.equals(expected.global_var)
    assert not earlier_edit.driver_var.equals(expected.driver_var)


if __name__ == "__main__":
    test_target_grid_scramble()
    print("grid variance leak test passed")
