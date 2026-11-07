"""E154: reset-year in-season pace blend, early rounds. Pre-weekend only, no 2026 rows."""
import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.common import table
from f1.features import pre_weekend_cols


def pace3(tr, te, F, **kw):
    s = 0.7 * te.t3_pace_gap + 0.3 * te.d3_pace_gap
    return s.fillna(s.max() if s.notna().any() else 0).values   # harness convention: LOWER score = better (pace gap: smaller = faster)


def main():
    X = table(); X = X[X.season <= 2025]
    base = pre_weekend_cols(X)
    W.MODELS["standings"] = lambda tr, te, f, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
    W.MODELS["logit"] = lambda tr, te, f, **kw: W.m_logit_pod(tr, te, base, C=0.03, hl=100)
    W.MODELS["pace3"] = pace3
    R, _ = W.walk_forward(X, range(2020, 2026), ["standings", "logit", "pace3"],
                          ens={"ens": ["standings", "logit"], "blend": ["standings", "pace3"],
                               "ens_pace": ["standings", "logit", "pace3"]}, tie=None)
    R["stage"] = np.where(R["round"] <= 10, "R1-10", "R11+")
    print(R.groupby(["season", "stage", "model"]).podium.mean().unstack().round(3).to_string())
    print("\nraces per cell:\n", R[R.model == "ens"].groupby(["season", "stage"]).size().unstack().to_string())
    for s in range(2020, 2026):
        z = R[(R.season == s) & (R.stage == "R1-10")].pivot(index="round", columns="model", values="podium")
        for c in ("blend", "ens_pace", "pace3", "ens"):
            d, ci, p = W.paired_boot(z[c], z["standings"])
            print(s, "R1-10", c, "minus standings", round(d, 3), np.round(ci, 3), "n", len(z))
    z = R[R.stage == "R1-10"].pivot(index=["season", "round"], columns="model", values="podium")
    nr = z[z.index.get_level_values(0) != 2022]
    for c in ("blend", "ens_pace"):
        print("placebo non-reset R1-10", c, "minus standings", W.paired_boot(nr[c], nr["standings"])[:2])


if __name__ == "__main__":
    main()
