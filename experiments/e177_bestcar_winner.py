"""E177: past team-race minima rather than two-car means for winner selection."""
import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.e168_winner import win_score
from f1.data import DATA, load_results
from f1.features import _prep, _wmean, build_features, pre_weekend_cols

INDICATORS = ["order", "q_gap", "pace_gap"]
BEST_FEATURES = [f"prior_bestcar{hl}_{col}" for hl in (3, 10) for col in INDICATORS]


def add_bestcar_history(X, source):
    teams = source.groupby(["k", "team"], sort=True).agg({"era": "first", **{col: "min" for col in INDICATORS}}).reset_index()
    out = X.copy()
    for k, cur in X.groupby("k", sort=True):
        past = teams[teams.k < k]
        for hl in (3, 10):
            weights = 0.5 ** ((k - past.k) / hl) * np.where(past.era < cur.era.iloc[0], .1, 1.)
            means = _wmean(past, ["team"], INDICATORS, weights)
            for col in INDICATORS:
                out.loc[cur.index, f"prior_bestcar{hl}_{col}"] = cur.team.map(means[col])
    return out


def main():
    df = load_results().loc[lambda d: d.season <= 2025]
    pace = pd.read_csv(DATA / "racepace_ff1.csv").loc[lambda d: d.season <= 2025]
    X = build_features(df, pace); base = pre_weekend_cols(X)
    X = add_bestcar_history(X, _prep(df, pace)); augmented = base + BEST_FEATURES
    W.assert_pre_weekend(augmented)
    W.MODELS.update(
        standings=lambda tr, te, F: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).to_numpy(),
        pre_logit=lambda tr, te, F: W.m_logit_pod(tr, te, base, C=.03, hl=100),
        win_logit=lambda tr, te, F: win_score(tr, te, base),
        bestcar_win=lambda tr, te, F: win_score(tr, te, augmented),
        bestcar_null=lambda tr, te, F: win_score(tr, te, augmented, shuffled=True),
    )
    R, P = W.walk_forward(X, list(range(2020, 2026)), ["standings", "pre_logit", "win_logit", "bestcar_win", "bestcar_null"],
                          F=augmented, ens={"ens": ["standings", "pre_logit"]}, tie=None)
    selected = P.sort_values(["score", "driver"]).drop_duplicates(["season", "round", "model"])
    selected = selected.merge(X[["season", "round", "driver", "team"]], on=["season", "round", "driver"], validate="many_to_one")
    truth = X.loc[X.win.eq(1), ["season", "round", "team"]]
    selected = selected.merge(truth, on=["season", "round"], suffixes=("_pred", "_true"))
    selected["team_hit"] = selected.team_pred.eq(selected.team_true)
    for split, years in (("inner", [2020, 2021, 2022]), ("outer", [2023, 2024, 2025])):
        r = R[R.season.isin(years)]
        print(split, r.groupby("model")[["winner", "podium"]].agg(["sum", "mean"]).round(4).to_string(), flush=True)
        print(r.pivot_table(index="season", columns="model", values="winner", aggfunc="sum").astype(int).to_string(), flush=True)
        print("winning-team hits", selected[selected.season.isin(years)].groupby("model").team_hit.sum().to_dict(), flush=True)
        a = r[r.model.eq("bestcar_win")].set_index(["season", "round"]).winner
        for ref in ("standings", "ens", "win_logit"):
            b = r[r.model.eq(ref)].set_index(["season", "round"]).winner
            print("bestcar_win minus", ref, W.paired_boot(a, b), flush=True)
    R.to_csv("results/e177_per_race.csv", index=False)
    P.to_csv("results/e177_predictions.csv", index=False)


if __name__ == "__main__":
    main()
