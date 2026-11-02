"""A2 prior-race pace cannot use target or later timing summaries."""
import pandas as pd

from experiments.a2_matchedpace import PART, prior_forecasts
from experiments.common import table


def test_matchedpace_uses_only_past():
    rows = [(2022, 1, 0, "AAA"), (2022, 1, 0, "BBB"),
            (2022, 2, 1, "AAA"), (2022, 2, 1, "BBB"),
            (2022, 3, 2, "AAA"), (2022, 3, 2, "BBB")]
    X = pd.DataFrame(rows, columns=["season", "round", "k", "code"])
    p = X[["season", "round", "code"]].assign(matched_gap=[0, 1, 2, 3, 4, 5], pace_gap=[1, 2, 3, 4, 5, 6])
    base = prior_forecasts(X, p)
    changed = p.copy()
    changed.loc[changed["round"] >= 2, ["matched_gap", "pace_gap"]] = 99
    got = prior_forecasts(X, changed)
    cols = ["matched_forecast", "median_forecast"]
    pd.testing.assert_frame_equal(base[base["round"] == 2][cols].reset_index(drop=True),
                                  got[got["round"] == 2][cols].reset_index(drop=True), check_exact=True)
    assert base[base["round"] == 2][cols].notna().all().all()
    changed.loc[changed["round"] == 1, ["matched_gap", "pace_gap"]] = 50
    got = prior_forecasts(X, changed)
    assert not base[base["round"] == 2][cols].reset_index(drop=True).equals(
        got[got["round"] == 2][cols].reset_index(drop=True))


def test_cached_source_ignores_target_and_later():
    X = table(); X = X[X.season <= 2025].copy()
    pace = pd.read_csv(PART)
    base = prior_forecasts(X, pace)
    for year, round_ in ((2023, 7), (2025, 20)):
        changed = pace.copy()
        later = (changed.season > year) | ((changed.season == year) & (changed["round"] >= round_))
        changed.loc[later, ["matched_gap", "pace_gap"]] = 999
        got = prior_forecasts(X, changed)
        target = lambda d: d[(d.season == year) & (d["round"] == round_)].sort_values("code")[["matched_forecast", "median_forecast"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(target(base), target(got), check_exact=True)


if __name__ == "__main__":
    test_matchedpace_uses_only_past()
    test_cached_source_ignores_target_and_later()
    print("matched pace leak test passed")
