"""The grid-distribution podium vote ignores the target race's actual grid."""

import pandas as pd

from experiments.b8_grid_distribution import add_grid_probability


def test_target_grid_scramble():
    source = pd.DataFrame([
        dict(season=season, round=1, k=k, driver=f"D{i}", code=f"D{i}",
             d_season_pts=6 - i, d10_order=i + 1,
             prior_grid=i + 1.5, gridn=i + 1)
        for season, k in ((2019, 1), (2020, 2)) for i in range(6)
    ])
    expected = add_grid_probability(source).query("season == 2020").grid_top6_p.reset_index(drop=True)
    changed = source.copy()
    changed.loc[changed.season == 2020, "gridn"] = 21
    actual = add_grid_probability(changed).query("season == 2020").grid_top6_p.reset_index(drop=True)
    pd.testing.assert_series_equal(expected, actual)
    changed.loc[changed.season == 2019, "gridn"] = 21
    earlier_edit = add_grid_probability(changed).query("season == 2020").grid_top6_p.reset_index(drop=True)
    assert not expected.equals(earlier_edit)


if __name__ == "__main__":
    test_target_grid_scramble()
    print("grid distribution leak test passed")
