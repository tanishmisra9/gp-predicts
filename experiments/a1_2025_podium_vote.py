"""E141: one fixed podium rank vote from E140's 2024 team X/Y profiles."""

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.a1_2025_team_transfer import PART, SESSIONS, prior_team_speed
from experiments.common import table
from f1.data import load_results
from f1.features import pre_weekend_cols


def main():
    X = table()
    X = X[X.season <= 2025].copy()
    results = load_results().query("season <= 2025")
    circuits = results[["season", "round", "circuit"]].drop_duplicates()
    X = X.merge(circuits, on=["season", "round"], validate="many_to_one")
    prior = prior_team_speed(pd.read_csv(PART), results)
    prior["season"] = 2025
    X = X.merge(prior[["season", "circuit", "team", "prior_xy_speed"]],
                on=["season", "circuit", "team"], how="left", validate="many_to_one")
    selected = X[(X.season == 2025) & X.circuit.isin(SESSIONS)]
    assert selected.prior_xy_speed.notna().all()
    base = pre_weekend_cols(X)
    W.assert_pre_weekend(base + ["prior_xy_speed"])
    W.MODELS["standings"] = lambda tr, te, F, **kw: (
        -te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
    W.MODELS["logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, base, C=0.03, hl=100)
    W.MODELS["xy_team"] = lambda tr, te, F, **kw: -te.prior_xy_speed.fillna(0).values

    def shuffled(tr, te, F, **kw):
        t = tr.copy()
        t["podium"] = np.random.default_rng(0).permutation(t.podium.values)
        return W.m_logit_pod(t, te, base, C=0.03, hl=100)

    W.MODELS["null_shuffled_labels"] = shuffled
    R, P = W.walk_forward(X, [2025], ["standings", "logit", "xy_team", "null_shuffled_labels"],
                          F=base, tie=None, ens={"ens": ["standings", "logit"],
                                                 "ens_xy": ["standings", "logit", "xy_team"]})
    rounds = set(selected["round"])
    R = R[R["round"].isin(rounds)].copy()
    P = P[P["round"].isin(rounds)].copy()
    z = R.pivot(index=["season", "round"], columns="model", values="podium")
    print("strict podium overlap", z.mean().round(4).to_dict(), "races", len(z))
    print("by race", z[["standings", "ens", "xy_team", "ens_xy", "null_shuffled_labels"]].round(4).to_string())
    for reference in ("standings", "ens"):
        print("ens_xy minus", reference, "paired", W.paired_boot(z.ens_xy, z[reference])[:2])
    picks = (P.sort_values(["model", "season", "round", "score", "driver"])
             .groupby(["model", "season", "round"]).head(3)
             .groupby(["model", "season", "round"]).driver.apply(frozenset))
    print("changed top-three versus ens", sum(picks["ens", y, r] != picks["ens_xy", y, r] for y, r in z.index))
    for y, r in z.index:
        print(y, r, "standings", sorted(picks["standings", y, r]),
              "ens", sorted(picks["ens", y, r]), "ens_xy", sorted(picks["ens_xy", y, r]))


if __name__ == "__main__":
    main()
