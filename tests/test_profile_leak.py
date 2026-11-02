"""Profile-feature leak proof: replace ALL telemetry / session / strategy / circuit rows of the target weekend and every later one with noise,
wipe its results, and assert every profile feature of the target race is bit-identical."""
import numpy as np
import pandas as pd

from f1.data import DATA, load_results
from f1.features import build_features, pre_weekend_cols
from f1.profile import PROFILE_COLS, add_profile, load_aux

WEEK = ["grid", "q_pos", "q1", "q2", "q3", "s_grid", "s_order", "s_points", "finish", "order", "points", "laps"]


def test_profile_features_ignore_target_weekend():
    df = load_results(); df = df[df.season >= 2022].reset_index(drop=True)
    pace = pd.read_csv(DATA / "racepace_ff1.csv"); aux = load_aux()
    X0 = build_features(df, pace); base = add_profile(X0, df, aux); F = pre_weekend_cols(X0) + PROFILE_COLS(base)
    assert any(c.startswith("fit_car") for c in F) and "stops_med" in F and "c_chaos" in F
    rng = np.random.default_rng(3)
    for tgt in [(2023, 7), (2024, 15), (2025, 20), (2026, 12), (2026, 18)]:
        late = lambda d: (d.season > tgt[0]) | ((d.season == tgt[0]) & (d["round"] >= tgt[1]))
        bad = df.copy(); m = late(bad)
        bad.loc[m, WEEK] = rng.random((m.sum(), len(WEEK))) * 20 + 1; bad.loc[m, "status"] = "Collision"; bad.loc[m, "grid"] = 1
        badaux = []
        for t in aux:
            t = t.copy(); mm = late(t); num = t.select_dtypes("number").columns.difference(["season", "round"])
            t.loc[mm, num] = rng.random((mm.sum(), len(num))) * 50; badaux.append(t)
        got = add_profile(build_features(bad, pace), bad, tuple(badaux))
        sel = lambda x: x[(x.season == tgt[0]) & (x["round"] == tgt[1])].sort_values("driver")[F].reset_index(drop=True)
        pd.testing.assert_frame_equal(sel(base), sel(got), check_exact=True)
        print("profile features clean at", tgt)


if __name__ == "__main__":
    test_profile_features_ignore_target_weekend()
