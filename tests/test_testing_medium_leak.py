"""A6 pre-season testing features ignore race-weekend labels and later seasons."""
import pandas as pd

from experiments.a6_testing_medium import PART, attach_testing
from experiments.common import table
from f1.data import DATA


def test_testing_feature_ignores_target_weekend():
    X = table(); X = X[X.season <= 2025].copy()
    medium = pd.read_csv(PART)
    best = pd.concat([pd.read_csv(p) for p in (DATA / "parts").glob("testing_*.csv")], ignore_index=True)
    best = best[best.season <= 2025]
    original = attach_testing(X, medium, best)
    for year, round_ in ((2023, 1), (2024, 2), (2025, 3)):
        changed = X.copy()
        late = (changed.season > year) | ((changed.season == year) & (changed["round"] >= round_))
        changed.loc[late, ["order", "podium", "win", "gridn", "q_pos"]] = 99
        got = attach_testing(changed, medium, best)
        target = lambda d: d[(d.season == year) & (d["round"] == round_)].sort_values("code")[["test_medium", "test_best"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(target(original), target(got), check_exact=True)
    changed = medium.copy()
    changed.loc[changed.season > 2023, "medium_gap"] = 999
    got = attach_testing(X, changed, best)
    target = lambda d: d[(d.season == 2023) & (d["round"] == 1)].sort_values("code").test_medium.reset_index(drop=True)
    pd.testing.assert_series_equal(target(original), target(got), check_exact=True)


if __name__ == "__main__":
    test_testing_feature_ignores_target_weekend()
    print("testing medium leak test passed")
