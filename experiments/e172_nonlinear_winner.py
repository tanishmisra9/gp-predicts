"""E172: fixed nonlinear winner models plus sensitivity to random-winner seed."""
from functools import partial

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.e168_winner import win_score
from experiments.e171_win_history import WIN_FEATURES, add_win_history
from f1.data import DATA, load_results
from f1.features import build_features, pre_weekend_cols


def winner_gbm(tr, te, F, shuffled=False):
    W.assert_pre_weekend(F)
    assert tr.k.max() < te.k.min()
    y = tr.win.to_numpy().copy()
    if shuffled:
        y[:] = 0; rng = np.random.default_rng(168)
        for ids in tr.groupby("k", sort=True).indices.values():
            y[rng.choice(ids)] = 1
    return W.m_lgb_pod(tr.assign(podium=y), te, F, hl=100)


def main():
    df = load_results().loc[lambda d: d.season <= 2025]
    pace = pd.read_csv(DATA / "racepace_ff1.csv").loc[lambda d: d.season <= 2025]
    X = add_win_history(build_features(df, pace))
    F = pre_weekend_cols(X); augmented = F + WIN_FEATURES
    W.MODELS.update(
        gbm_win=lambda tr, te, F: winner_gbm(tr, te, F),
        gbm_history=lambda tr, te, F: winner_gbm(tr, te, augmented),
        gbm_null=lambda tr, te, F: winner_gbm(tr, te, augmented, shuffled=True),
    )
    R, P = W.walk_forward(X, list(range(2020, 2026)), ["gbm_win", "gbm_history", "gbm_null"], F=F, tie=None)
    ref = pd.read_csv("results/e171_per_race.csv")
    assert set(map(tuple, ref[["season", "round"]].to_numpy())) == set(map(tuple, R[["season", "round"]].to_numpy()))
    R = pd.concat([R, ref[ref.model.isin(["standings", "ens", "win_logit", "win_history"])]]).reset_index(drop=True)
    picked = P.sort_values(["score", "driver"]).drop_duplicates(["season", "round", "model"])
    truth = X.loc[X.win.eq(1), ["season", "round", "driver"]]
    checked = picked.merge(truth, on=["season", "round"], suffixes=("_pred", "_true"))
    hits = checked.assign(hit=checked.driver_pred.eq(checked.driver_true)).groupby("model").hit.sum()
    assert hits.eq(R[R.model.isin(hits.index)].groupby("model").winner.sum()).all()
    for split, years in (("inner", [2020, 2021, 2022]), ("outer", [2023, 2024, 2025])):
        d = R[R.season.isin(years)]
        print(split, d.groupby("model")[["winner", "podium"]].agg(["sum", "mean"]).round(4).to_string(), flush=True)
        print(d.pivot_table(index="season", columns="model", values="winner", aggfunc="sum").astype(int).to_string(), flush=True)
        for candidate in ("gbm_win", "gbm_history"):
            a = d[d.model.eq(candidate)].set_index(["season", "round"]).winner
            for baseline in ("standings", "ens", "win_logit"):
                b = d[d.model.eq(baseline)].set_index(["season", "round"]).winner
                print(candidate, "minus", baseline, W.paired_boot(a, b), flush=True)
    R.to_csv("results/e172_per_race.csv", index=False)
    P.to_csv("results/e172_predictions.csv", index=False)
    nulls = []
    for seed in range(20):
        W.MODELS["seed_null"] = partial(win_score, shuffled=True, seed=seed)
        N, _ = W.walk_forward(X, list(range(2020, 2026)), ["seed_null"], F=augmented, tie=None)
        nulls.append(N.assign(seed=seed))
        print("null seed", seed, "finished", flush=True)
    N = pd.concat(nulls, ignore_index=True)
    N.to_csv("results/e172_null_audit.csv", index=False)
    for split, years in (("inner", [2020, 2021, 2022]), ("outer", [2023, 2024, 2025])):
        rates = N[N.season.isin(years)].groupby("seed").winner.mean()
        observed = ref[ref.season.isin(years) & ref.model.eq("win_history")].winner.mean()
        print(split, "null rates", rates.to_dict(), "mean", rates.mean(), "range", [rates.min(), rates.max()],
              "at least observed", int(rates.ge(observed).sum()), "of 20", flush=True)


if __name__ == "__main__":
    main()
