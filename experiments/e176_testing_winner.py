"""E176: fixed pre-season testing rules for winner choice in rounds 1–3."""
import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.e168_winner import win_score
from f1.data import DATA, load_results
from f1.features import build_features, pre_weekend_cols

TESTING_FILES = [DATA / "parts/testing_2022_2023.csv", DATA / "parts/testing_2024_2025.csv"]


def attach_testing(X, source):
    summary = source.groupby(["season", "code"])[["best_gap", "lr_gap"]].mean()
    out = X.join(summary, on=["season", "code"])
    groups = ["season", "round", "team"]
    for col in ("best_gap", "lr_gap"):
        out[col] = out.groupby(groups)[col].transform("mean")
    out["prior_testing_seen"] = out.groupby(["season", "round"]).best_gap.transform("count").gt(0)
    ranks = out.groupby(["season", "round"])[["best_gap", "lr_gap"]].rank(pct=True).fillna(.5)
    out["prior_testing_strength"] = ranks.mean(axis=1)
    return out.drop(columns=["best_gap", "lr_gap"])


def testing_score(te, base_score, team_first=False):
    if te["round"].iloc[0] > 3 or not te.prior_testing_seen.any():
        return np.asarray(base_score)
    strength = te.prior_testing_strength.to_numpy()
    ranked = W._rank01(base_score)
    return strength * 100 + ranked if team_first else .75 * ranked + .25 * strength


def main():
    df = load_results().loc[lambda d: d.season <= 2025]
    pace = pd.read_csv(DATA / "racepace_ff1.csv").loc[lambda d: d.season <= 2025]
    X = build_features(df, pace); base = pre_weekend_cols(X)
    source = pd.concat([pd.read_csv(p) for p in TESTING_FILES], ignore_index=True)
    X = attach_testing(X, source)
    W.assert_pre_weekend(base + ["prior_testing_strength", "prior_testing_seen"])
    W.MODELS.update(
        standings=lambda tr, te, F: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).to_numpy(),
        pre_logit=lambda tr, te, F: W.m_logit_pod(tr, te, base, C=.03, hl=100),
        win_logit=win_score,
        testing_blend=lambda tr, te, F: testing_score(te, win_score(tr, te, base)),
        testing_first=lambda tr, te, F: testing_score(te, win_score(tr, te, base), team_first=True),
        testing_null=lambda tr, te, F: testing_score(te, win_score(tr, te, base, shuffled=True)),
    )
    R, P = W.walk_forward(X, list(range(2020, 2026)),
                          ["standings", "pre_logit", "win_logit", "testing_blend", "testing_first", "testing_null"],
                          F=base, ens={"ens": ["standings", "pre_logit"]}, tie=None)
    coverage = X[["season", "round", "prior_testing_seen"]].drop_duplicates()
    R = R.merge(coverage, on=["season", "round"], validate="many_to_one")
    R["covered"] = R.prior_testing_seen & R["round"].le(3)
    for split, years in (("inner", [2020, 2021, 2022]), ("outer", [2023, 2024, 2025])):
        r = R[R.season.isin(years)]
        print(split, "all", r.groupby("model")[["winner", "podium"]].agg(["sum", "mean"]).round(4).to_string(), flush=True)
        print(split, "covered", r[r.covered].groupby("model")[["winner", "podium"]].agg(["sum", "mean"]).round(4).to_string(), flush=True)
        print(r[r.covered].pivot_table(index="season", columns="model", values="winner", aggfunc="sum").astype(int).to_string(), flush=True)
        for name in ("testing_blend", "testing_first"):
            a = r[r.model.eq(name)].set_index(["season", "round"]).winner
            for ref in ("standings", "ens", "win_logit"):
                b = r[r.model.eq(ref)].set_index(["season", "round"]).winner
                print(name, "minus", ref, W.paired_boot(a, b), flush=True)
    fallback = P.merge(coverage, on=["season", "round"], validate="many_to_one")
    fallback = fallback[(fallback["round"] > 3) | ~fallback.prior_testing_seen]
    pivot = fallback.pivot(index=["season", "round", "driver"], columns="model", values="score")
    np.testing.assert_array_equal(pivot.testing_blend, pivot.win_logit)
    np.testing.assert_array_equal(pivot.testing_first, pivot.win_logit)
    R.to_csv("results/e176_per_race.csv", index=False)
    P.to_csv("results/e176_predictions.csv", index=False)


if __name__ == "__main__":
    main()
