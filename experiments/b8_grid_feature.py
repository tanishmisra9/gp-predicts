"""E115: add an out-of-fold pre-weekend grid forecast to the podium logit."""

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.b8_grid_forecast import grid_forecast
from experiments.common import table
from f1.features import pre_weekend_cols


def add_prior_grid(X):
    """For each 2019+ row, fit the grid forecast only on earlier races."""
    features = pre_weekend_cols(X)
    W.assert_pre_weekend(features)
    out = X.copy()
    out["prior_grid"] = np.nan
    for _, te in out[out.season >= 2019].groupby("k", sort=True):
        tr = out[(out.k < te.k.iloc[0]) & (out.season >= 2018)]
        if len(tr):
            out.loc[te.index, "prior_grid"] = grid_forecast(tr, te, features)
    return out


def main():
    X = table(); X = X[X.season <= 2025].copy()
    base = pre_weekend_cols(X)
    X = add_prior_grid(X)
    W.assert_pre_weekend(base + ["prior_grid"])
    W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
    W.MODELS["logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, base, C=0.03, hl=100)
    W.MODELS["logit_grid"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, base + ["prior_grid"], C=0.03, hl=100)

    def shuffled(tr, te, F, **kw):
        t = tr.copy()
        t["podium"] = np.random.default_rng(0).permutation(t.podium.values)
        return W.m_logit_pod(t, te, base, C=0.03, hl=100)

    W.MODELS["null_shuffled_labels"] = shuffled
    R, P = W.walk_forward(X, range(2020, 2026),
                          ["standings", "logit", "logit_grid", "null_shuffled_labels"],
                          ens={"ens": ["standings", "logit"],
                               "ens_grid_feature": ["standings", "logit_grid"]}, tie=None)
    R["split"] = np.where(R.season <= 2022, "inner", "outer")
    print("strict overlap", R.groupby(["model", "split"]).podium.mean().unstack().round(4).to_string())
    picks = P[P.model.isin(["ens", "ens_grid_feature"])].sort_values(["model", "season", "round", "score", "driver"])
    picks = picks.groupby(["model", "season", "round"]).head(3)
    picks = picks.groupby(["model", "season", "round"]).driver.apply(frozenset)
    for split in ("inner", "outer"):
        z = R[R.split == split].pivot(index=["season", "round"], columns="model", values="podium")
        changed = sum(picks["ens", y, r] != picks["ens_grid_feature", y, r] for y, r in z.index)
        print(split, "top3 changes", changed, "feature ensemble vs standings", W.paired_boot(z.ens_grid_feature, z.standings)[:2],
              "vs ens", W.paired_boot(z.ens_grid_feature, z.ens)[:2])
    p = P[P.model.isin(["logit", "logit_grid"])].merge(
        X[["season", "round", "driver", "podium"]], on=["season", "round", "driver"], validate="many_to_one")
    p["prob"] = (-p.score).clip(1e-6, 1 - 1e-6)
    p["brier"] = (p.prob - p.podium) ** 2
    p["logloss"] = -(p.podium * np.log(p.prob) + (1 - p.podium) * np.log1p(-p.prob))
    p["split"] = np.where(p.season <= 2022, "inner", "outer")
    print("probabilities", p.groupby(["model", "split"])[["brier", "logloss"]].mean().round(5).to_string())
    by_race = p.groupby(["split", "season", "round", "model"])[["brier", "logloss"]].mean().reset_index()
    for split in ("inner", "outer"):
        z = by_race[by_race.split == split].pivot(index=["season", "round"], columns="model", values=["brier", "logloss"])
        for metric in ("brier", "logloss"):
            print(split, metric, "augmented minus frozen", W.paired_boot(z[metric].logit_grid, z[metric].logit)[:2])


if __name__ == "__main__":
    main()
