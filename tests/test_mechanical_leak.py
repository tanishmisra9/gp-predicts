"""A4 history features must ignore the target race and every later status/lap row."""
import pandas as pd

from experiments.a4_mechanical import NEW, add_mechanical_history
from experiments.common import table
from f1.data import load_results


def test_mechanical_history_ignores_target_and_later():
    X = table(); X = X[X.season <= 2025].copy()
    results = load_results(); results = results[results.season <= 2025].copy()
    original = add_mechanical_history(X, results)
    for year, round_ in ((2022, 10), (2023, 7), (2025, 20)):
        changed = results.copy()
        m = (changed.season > year) | ((changed.season == year) & (changed["round"] >= round_))
        changed.loc[m, "status"] = "Engine"
        changed.loc[m, "laps"] = 1
        got = add_mechanical_history(X, changed)
        target = lambda d: d[(d.season == year) & (d["round"] == round_)].sort_values("code")[NEW].reset_index(drop=True)
        pd.testing.assert_frame_equal(target(original), target(got), check_exact=True)
    # Mutation check: a prior result must affect at least one later feature.
    changed = results.copy()
    prior = (changed.season == 2022) & (changed["round"] == 9)
    changed.loc[prior, "status"] = "Engine"
    changed.loc[prior, "laps"] = 1
    got = add_mechanical_history(X, changed)
    target = lambda d: d[(d.season == 2022) & (d["round"] == 10)][NEW].reset_index(drop=True)
    assert not target(original).equals(target(got))


if __name__ == "__main__":
    test_mechanical_history_ignores_target_and_later()
    print("mechanical history leak test passed")
