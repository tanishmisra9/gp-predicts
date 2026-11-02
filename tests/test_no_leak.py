"""Leak proof: scrambling every post-race outcome (results AND race-lap pace) of race K and all later races
must not change race K's pre-race features. Pre-race inputs (grid, quali, sprint, practice laps) stay untouched."""
import numpy as np
import pandas as pd

from f1.data import DATA, load_results
from f1.features import build_features

LABELS = {"order", "podium", "win"}


def test_features_ignore_current_and_future_outcomes():
    df = load_results()
    df = df[df.season >= 2024].reset_index(drop=True)
    laps = pd.read_csv(DATA / "laps.csv")
    rng = np.random.default_rng(0)
    base = build_features(df, laps)
    feats = [c for c in base if c not in LABELS]
    assert any(c.startswith("lr_gap_") for c in feats) and any(c.startswith("d10_pace_gap") for c in feats)
    for target in [(2024, 7), (2025, 12), (2026, 10), (2026, 19)]:
        bad, badl = df.copy(), laps.copy()
        m = (bad.season > target[0]) | ((bad.season == target[0]) & (bad["round"] >= target[1]))
        bad.loc[m, "order"] = rng.integers(1, 23, m.sum())
        bad.loc[m, "finish"] = rng.integers(1, 23, m.sum())
        bad.loc[m, "points"] = rng.integers(0, 26, m.sum())
        bad.loc[m, "laps"] = rng.integers(0, 70, m.sum())
        bad.loc[m, "status"] = rng.choice(["Finished", "Engine", "Collision", "+1 Lap"], m.sum())
        ml = (badl.session == "R") & ((badl.season > target[0]) | ((badl.season == target[0]) & (badl["round"] >= target[1])))
        badl.loc[ml, "pace_gap"] = rng.random(ml.sum()) * 3
        got = build_features(bad, badl)
        sel = lambda x: x[(x.season == target[0]) & (x["round"] == target[1])][feats].reset_index(drop=True)
        pd.testing.assert_frame_equal(sel(base), sel(got), check_exact=True)
        print("no leak at", target)


if __name__ == "__main__":
    test_features_ignore_current_and_future_outcomes()
