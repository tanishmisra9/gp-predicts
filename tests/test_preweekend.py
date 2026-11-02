"""Pre-weekend leak proof: wiping ALL data of the target weekend (grid, quali, sprint, practice laps, and outcomes) must not
change the pre-weekend features of that race. Only the entry list (driver, team, circuit) of the target weekend remains."""
import numpy as np
import pandas as pd

from f1.data import DATA, load_results
from f1.features import build_features, pre_weekend_cols

WEEKEND = ["grid", "q_pos", "q1", "q2", "q3", "s_grid", "s_order", "s_points"]
OUT = ["order", "finish", "points", "laps"]


def test_preweekend_features_ignore_target_weekend():
    df = load_results(); df = df[df.season >= 2023].reset_index(drop=True)
    pace = pd.read_csv(DATA / "racepace_ff1.csv")
    base = build_features(df, pace); F = pre_weekend_cols(base)
    assert not {"gridn", "q_gap", "q_pos", "grid_pen", "s_order"} & set(F)
    for tgt in [(2023, 6), (2024, 12), (2025, 20), (2026, 10), (2026, 18)]:
        bad, bp = df.copy(), pace.copy()
        m = (bad.season > tgt[0]) | ((bad.season == tgt[0]) & (bad["round"] >= tgt[1]))
        bad.loc[m, WEEKEND] = np.random.default_rng(1).random((m.sum(), len(WEEKEND))) * 20 + 1
        bad.loc[m, OUT] = np.random.default_rng(2).integers(1, 23, (m.sum(), len(OUT)))
        bad.loc[m, "status"] = "Engine"
        bp.loc[(bp.season > tgt[0]) | ((bp.season == tgt[0]) & (bp["round"] >= tgt[1])), "pace_gap"] = 7.7
        got = build_features(bad, bp)
        sel = lambda x: x[(x.season == tgt[0]) & (x["round"] == tgt[1])][F].reset_index(drop=True)
        pd.testing.assert_frame_equal(sel(base), sel(got), check_exact=True)
        print("pre-weekend features clean at", tgt)


if __name__ == "__main__":
    test_preweekend_features_ignore_target_weekend()
