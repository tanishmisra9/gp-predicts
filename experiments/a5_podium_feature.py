"""E109: one matched-teammate pace feature in the frozen pre-weekend logit."""

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.a5_teammate_segments import PART, prior_pair
from experiments.common import table
from f1.features import pre_weekend_cols


def feature_table(X, source):
    entrants = X[["season", "round", "k", "team", "code"]]
    mates = entrants.merge(entrants, on=["season", "round", "k", "team"], suffixes=("", "_mate"))
    pairs = mates[mates.code != mates.code_mate].rename(columns={"code_mate": "mate"})
    f = prior_pair(pairs, source)
    return X.merge(f[["season", "round", "code", "prior_gap"]],
                   on=["season", "round", "code"], how="left", validate="one_to_one")


def main():
    X = table()
    X = X[X.season <= 2025].copy()
    base = pre_weekend_cols(X)
    source = pd.read_csv(PART)
    X = feature_table(X, source)
    W.assert_pre_weekend(base + ["prior_gap"])
    W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
    W.MODELS["logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, base, C=0.03, hl=100)
    W.MODELS["logit_pair"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, base + ["prior_gap"], C=0.03, hl=100)
    def shuffled(tr, te, F, **kw):
        t = tr.copy()
        t["podium"] = np.random.default_rng(0).permutation(t.podium.values)
        return W.m_logit_pod(t, te, base, C=0.03, hl=100)
    W.MODELS["null_shuffled_labels"] = shuffled
    R, P = W.walk_forward(X, range(2020, 2026),
                          ["standings", "logit", "logit_pair", "null_shuffled_labels"],
                          ens={"ens": ["standings", "logit"],
                               "ens_pair": ["standings", "logit_pair"]}, tie=None)
    R["split"] = np.where(R.season <= 2022, "inner", "outer")
    print("strict overlap", R.groupby(["model", "split"]).podium.mean().unstack().round(4).to_string())
    picks = P[P.model.isin(["ens", "ens_pair"])].sort_values(["model", "season", "round", "score", "driver"])
    picks = picks.groupby(["model", "season", "round"]).head(3)
    picks = picks.groupby(["model", "season", "round"]).driver.apply(frozenset)
    for split in ("inner", "outer"):
        z = R[R.split == split].pivot(index=["season", "round"], columns="model", values="podium")
        changed = sum(picks["ens", year, round_] != picks["ens_pair", year, round_] for year, round_ in z.index)
        print(split, "coverage", round(X[X.season.isin(range(2020, 2023) if split == "inner" else range(2023, 2026))].prior_gap.notna().mean(), 4),
              "top3 changed", changed, "pair ens vs standings", W.paired_boot(z.ens_pair, z.standings)[:2],
              "vs ens", W.paired_boot(z.ens_pair, z.ens)[:2])
    p = P[P.model.isin(["logit", "logit_pair"])].merge(
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
            print(split, metric, "pair-logit minus frozen-logit race-paired",
                  W.paired_boot(z[metric]["logit_pair"], z[metric]["logit"])[:2])


if __name__ == "__main__":
    main()
