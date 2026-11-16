"""E173: one fixed shorter training half-life for the winner logistic."""
from functools import partial

import pandas as pd

import f1.walkforward as W
from experiments.e168_winner import win_score
from f1.data import DATA, load_results
from f1.features import build_features, pre_weekend_cols


def main():
    df = load_results().loc[lambda d: d.season <= 2025]
    pace = pd.read_csv(DATA / "racepace_ff1.csv").loc[lambda d: d.season <= 2025]
    X = build_features(df, pace); F = pre_weekend_cols(X)
    W.MODELS.update(recent_win=partial(win_score, hl=20), recent_null=partial(win_score, shuffled=True, hl=20))
    R, P = W.walk_forward(X, list(range(2020, 2026)), ["recent_win", "recent_null"], F=F, tie=None)
    reference = pd.read_csv("results/e169_per_race.csv")
    assert set(map(tuple, reference[["season", "round"]].to_numpy())) == set(map(tuple, R[["season", "round"]].to_numpy()))
    R = pd.concat([R, reference[reference.model.isin(["standings", "ens", "win_logit"])]]).reset_index(drop=True)
    for split, years in (("inner", [2020, 2021, 2022]), ("outer", [2023, 2024, 2025])):
        d = R[R.season.isin(years)]
        print(split, d.groupby("model")[["winner", "podium"]].agg(["sum", "mean"]).round(4).to_string(), flush=True)
        print(d.pivot_table(index="season", columns="model", values="winner", aggfunc="sum").astype(int).to_string(), flush=True)
        a = d[d.model.eq("recent_win")].set_index(["season", "round"]).winner
        for ref in ("standings", "ens", "win_logit"):
            b = d[d.model.eq(ref)].set_index(["season", "round"]).winner
            print("recent_win minus", ref, W.paired_boot(a, b), flush=True)
    picked = P.sort_values(["score", "driver"]).drop_duplicates(["season", "round", "model"])
    truth = X.loc[X.win.eq(1), ["season", "round", "driver"]]
    checked = picked.merge(truth, on=["season", "round"], suffixes=("_pred", "_true"))
    hits = checked.assign(hit=checked.driver_pred.eq(checked.driver_true)).groupby("model").hit.sum()
    assert hits.eq(R[R.model.isin(hits.index)].groupby("model").winner.sum()).all()
    R.to_csv("results/e173_per_race.csv", index=False)
    P.to_csv("results/e173_predictions.csv", index=False)


if __name__ == "__main__":
    main()
