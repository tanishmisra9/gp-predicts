"""E134: fixed podium vote from a past-only forecast-grid distribution."""

import numpy as np
import pandas as pd
from scipy.stats import norm

import f1.walkforward as W
from experiments.b8_grid_feature import add_prior_grid
from experiments.b8_grid_variance import prior_variance
from experiments.common import table
from f1.features import pre_weekend_cols


def add_grid_probability(X):
    variance = prior_variance(X)[["season", "round", "driver", "global_var", "driver_var"]]
    out = X.merge(variance, on=["season", "round", "driver"], how="left", validate="one_to_one")
    global_var = out.groupby(["season", "round"]).global_var.transform("max")
    scale = np.sqrt(out.driver_var.fillna(global_var))
    out["grid_top6_p"] = norm.cdf((6.5 - out.prior_grid) / scale)
    return out.drop(columns=["global_var", "driver_var"])


def main():
    X = table()
    X = X[X.season <= 2025].copy()
    base = pre_weekend_cols(X)
    X = add_grid_probability(add_prior_grid(X))
    assert X[X.season >= 2020].grid_top6_p.notna().all()
    W.assert_pre_weekend(base + ["prior_grid", "grid_top6_p"])
    W.MODELS["standings"] = lambda tr, te, F, **kw: (
        -te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
    W.MODELS["logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, base, C=0.03, hl=100)
    W.MODELS["grid_distribution"] = lambda tr, te, F, **kw: -te.grid_top6_p.values

    def shuffled(tr, te, F, **kw):
        t = tr.copy()
        t["podium"] = np.random.default_rng(0).permutation(t.podium.values)
        return W.m_logit_pod(t, te, base, C=0.03, hl=100)

    W.MODELS["null_shuffled_labels"] = shuffled
    R, P = W.walk_forward(X, range(2020, 2026),
                          ["standings", "logit", "grid_distribution", "null_shuffled_labels"],
                          F=base, tie=None,
                          ens={"ens": ["standings", "logit"],
                               "ens_grid_distribution": ["standings", "logit", "grid_distribution"]})
    R["split"] = np.where(R.season <= 2022, "inner", "outer")
    print("strict overlap", R.groupby(["model", "split"]).podium.mean().unstack().round(4).to_string())
    for split in ("inner", "outer"):
        z = R[R.split == split].pivot(index=["season", "round"], columns="model", values="podium")
        for ref in ("standings", "ens"):
            print(split, "ens_grid_distribution minus", ref,
                  W.paired_boot(z.ens_grid_distribution, z[ref])[:2])
    picks = P[P.model.isin(["ens", "ens_grid_distribution"])]
    picks = (picks.sort_values(["model", "season", "round", "score", "driver"])
             .groupby(["model", "season", "round"]).head(3)
             .groupby(["model", "season", "round"]).driver.apply(frozenset))
    for split, seasons in (("inner", range(2020, 2023)), ("outer", range(2023, 2026))):
        print(split, "changed top-three sets",
              sum(picks["ens", y, r] != picks["ens_grid_distribution", y, r]
                  for y, r in R[(R.model == "ens") & R.season.isin(seasons)][["season", "round"]].itertuples(index=False)))


if __name__ == "__main__":
    main()
