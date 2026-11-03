"""E114: can a past-only grid forecast add podium information?"""

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.common import reg, table
from f1.features import pre_weekend_cols


def grid_forecast(tr, te, features):
    """Predict the target grid from strictly earlier race grids and pre-weekend features."""
    return reg(tr, te, features, y="gridn", kind="ridge", alpha=300, hl=100)


def main():
    X = table(); X = X[X.season <= 2025].copy()
    features = pre_weekend_cols(X)
    W.assert_pre_weekend(features)
    W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
    W.MODELS["logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, features, C=0.03, hl=100)
    W.MODELS["forecast_grid"] = lambda tr, te, F, **kw: grid_forecast(tr, te, features)

    def shuffled(tr, te, F, **kw):
        t = tr.copy()
        t["podium"] = np.random.default_rng(0).permutation(t.podium.values)
        return W.m_logit_pod(t, te, features, C=0.03, hl=100)

    W.MODELS["null_shuffled_labels"] = shuffled
    models = ["standings", "logit", "forecast_grid", "null_shuffled_labels"]
    R, P = W.walk_forward(X, [2020, 2021, 2022, 2023, 2024, 2025], models, F=features,
                          ens={"ens": ["standings", "logit"],
                               "ens_grid": ["standings", "logit", "forecast_grid"]}, tie=None)
    R["split"] = np.where(R.season <= 2022, "inner", "outer")
    print("Strict podium overlap:")
    print(R.groupby(["model", "split"]).podium.mean().unstack().round(4).to_string())
    G = P.merge(X[["season", "round", "driver", "gridn"]], on=["season", "round", "driver"])
    contenders = P[P.model == "standings"].copy()
    contenders["prior_rank"] = contenders.groupby(["season", "round"]).score.rank(method="first")
    G6 = G.merge(contenders[["season", "round", "driver", "prior_rank"]],
                 on=["season", "round", "driver"])
    G6 = G6[G6.prior_rank <= 6]
    G6 = G6.groupby(["model", "season", "round"]).apply(
        lambda g: g.score.corr(g.gridn, method="spearman"), include_groups=False).rename("grid_rho").reset_index()
    G6["split"] = np.where(G6.season <= 2022, "inner", "outer")
    G = G.groupby(["model", "season", "round"]).apply(
        lambda g: g.score.corr(g.gridn, method="spearman"), include_groups=False).rename("grid_rho").reset_index()
    G["split"] = np.where(G.season <= 2022, "inner", "outer")
    print("Forecast versus actual grid-order Spearman:")
    print(G.groupby(["model", "split"]).grid_rho.mean().unstack().round(4).to_string())
    print("Same metric within standings' pre-weekend top six:")
    print(G6.groupby(["model", "split"]).grid_rho.mean().unstack().round(4).to_string())
    for split in ("inner", "outer"):
        d = R[R.split == split].pivot(index=["season", "round"], columns="model", values="podium")
        g = G[G.split == split].pivot(index=["season", "round"], columns="model", values="grid_rho")
        for candidate, reference in (("forecast_grid", "standings"), ("ens_grid", "ens"), ("ens_grid", "standings")):
            diff, ci, _ = W.paired_boot(d[candidate], d[reference])
            print(split, "podium", candidate, "vs", reference, round(diff, 4), np.round(ci, 4).tolist())
        diff, ci, _ = W.paired_boot(g.forecast_grid, g.standings)
        print(split, "grid rho forecast_grid vs standings", round(diff, 4), np.round(ci, 4).tolist())
        g6 = G6[G6.split == split].pivot(index=["season", "round"], columns="model", values="grid_rho").dropna()
        diff, ci, _ = W.paired_boot(g6.forecast_grid, g6.standings)
        print(split, "top-six grid rho forecast_grid vs standings", round(diff, 4), np.round(ci, 4).tolist(), "races", len(g6))


if __name__ == "__main__":
    main()
