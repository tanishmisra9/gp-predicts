"""E171: explicit strictly past driver/team win rates as winner-model features."""
from functools import partial

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.e168_winner import win_score
from f1.data import DATA, load_results
from f1.features import build_features, pre_weekend_cols

WIN_FEATURES = [f"prior_{who}_win{hl}" for who in ("driver", "team") for hl in (3, 10)]


def add_win_history(X):
    X = X.copy()
    teams = X.groupby(["k", "team"], sort=True).agg(win=("win", "max"), era=("era", "first")).reset_index()
    for k, cur in X.groupby("k", sort=True):
        for who, source, prior, era_weight in (("driver", X, 0.05, 0.3), ("team", teams, 0.10, 0.1)):
            past = source[source.k < k]
            for hl in (3, 10):
                w = 0.5 ** ((k - past.k) / hl) * np.where(past.era < cur.era.iloc[0], era_weight, 1.)
                numerator = (w * past.win).groupby(past[who]).sum()
                denominator = w.groupby(past[who]).sum()
                rate = (numerator + 3 * prior) / (denominator + 3)
                X.loc[cur.index, f"prior_{who}_win{hl}"] = cur[who].map(rate).fillna(prior)
    assert np.isfinite(X[WIN_FEATURES]).all().all()
    return X


def main():
    df = load_results().loc[lambda d: d.season <= 2025]
    pace = pd.read_csv(DATA / "racepace_ff1.csv").loc[lambda d: d.season <= 2025]
    X = add_win_history(build_features(df, pace))
    F = pre_weekend_cols(X) + WIN_FEATURES
    W.assert_pre_weekend(F)
    W.MODELS.update(
        win_history=win_score,
        history_null=partial(win_score, shuffled=True),
        rate3=lambda tr, te, F: -te.prior_driver_win3.to_numpy(),
        rate10=lambda tr, te, F: -te.prior_driver_win10.to_numpy(),
    )
    R, P = W.walk_forward(X, list(range(2020, 2026)), ["win_history", "history_null", "rate3", "rate10"], F=F, tie=None)
    reference = pd.read_csv("results/e169_per_race.csv")
    assert set(map(tuple, reference[["season", "round"]].to_numpy())) == set(map(tuple, R[["season", "round"]].to_numpy()))
    R = pd.concat([R, reference[reference.model.isin(["standings", "ens", "win_logit"])]])
    for split, years in (("inner", [2020, 2021, 2022]), ("outer", [2023, 2024, 2025])):
        d = R[R.season.isin(years)]
        print(split, d.groupby("model")[["winner", "podium"]].agg(["sum", "mean"]).round(4).to_string(), flush=True)
        print(d.pivot_table(index="season", columns="model", values="winner", aggfunc="sum").astype(int).to_string(), flush=True)
        a = d[d.model.eq("win_history")].set_index(["season", "round"]).winner
        for ref in ("standings", "ens", "win_logit"):
            b = d[d.model.eq(ref)].set_index(["season", "round"]).winner
            print("win_history minus", ref, W.paired_boot(a, b), flush=True)
    picked = P.sort_values(["score", "driver"]).drop_duplicates(["season", "round", "model"])
    truth = X.loc[X.win.eq(1), ["season", "round", "driver"]]
    checked = picked.merge(truth, on=["season", "round"], suffixes=("_pred", "_true"))
    hits = checked.assign(hit=checked.driver_pred.eq(checked.driver_true)).groupby("model").hit.sum()
    assert hits.eq(R[R.model.isin(hits.index)].groupby("model").winner.sum()).all()
    R.to_csv("results/e171_per_race.csv", index=False)
    P.to_csv("results/e171_predictions.csv", index=False)


if __name__ == "__main__":
    main()
