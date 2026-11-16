"""E175: the existing portable teammate pace graph as a winner-specific input."""
import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.a5_teammate_segments import PART
from experiments.b6_portable_teammate import feature_table
from experiments.e168_winner import win_score
from f1.data import DATA, load_results
from f1.features import build_features, pre_weekend_cols


def swap_winner(te, base_score):
    ranked = te.assign(base_score=np.asarray(base_score)).sort_values(["base_score", "driver"])
    team = ranked.iloc[0].team
    mates = ranked[ranked.team.eq(team)]
    if mates.prior_portable_gap.isna().any():
        return np.asarray(base_score)
    selected = mates.sort_values(["prior_portable_gap", "driver"]).index[0]
    ranks = pd.Series(np.arange(len(ranked), dtype=float), index=ranked.index)
    ranks.loc[selected] = -1.
    return ranks.reindex(te.index).to_numpy()


def main():
    df = load_results().loc[lambda d: d.season <= 2025]
    pace = pd.read_csv(DATA / "racepace_ff1.csv").loc[lambda d: d.season <= 2025]
    X = build_features(df, pace); base = pre_weekend_cols(X)
    source = pd.read_csv(PART).loc[lambda d: d.season <= 2025]
    X = feature_table(X, source)
    augmented = base + ["prior_portable_gap"]
    W.assert_pre_weekend(augmented)

    def standings(tr, te, F):
        return (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).to_numpy()

    def podium(tr, te, F):
        return W.m_logit_pod(tr, te, base, C=0.03, hl=100)

    def graph_swap(tr, te, F):
        score = (W._rank01(standings(tr, te, F)) + W._rank01(podium(tr, te, F))) / 2
        return swap_winner(te, score)

    W.MODELS.update(standings=standings, pre_logit=podium,
                    win_logit=lambda tr, te, F: win_score(tr, te, base),
                    portable_win=lambda tr, te, F: win_score(tr, te, augmented),
                    portable_null=lambda tr, te, F: win_score(tr, te, augmented, shuffled=True),
                    graph_swap=graph_swap)
    models = ["standings", "pre_logit", "win_logit", "portable_win", "portable_null", "graph_swap"]
    R, P = W.walk_forward(X, list(range(2020, 2026)), models, F=augmented,
                          ens={"ens": ["standings", "pre_logit"]}, tie=None)
    for split, years in (("inner", [2020, 2021, 2022]), ("outer", [2023, 2024, 2025])):
        r = R[R.season.isin(years)]
        print(split, r.groupby("model")[["winner", "podium"]].agg(["sum", "mean"]).round(4).to_string(), flush=True)
        print(r.pivot_table(index="season", columns="model", values="winner", aggfunc="sum").astype(int).to_string(), flush=True)
        print("graph coverage", X[X.season.isin(years)].prior_portable_gap.notna().mean(), flush=True)
        for name in ("portable_win", "graph_swap"):
            a = r[r.model.eq(name)].set_index(["season", "round"]).winner
            for ref in ("standings", "ens", "win_logit"):
                b = r[r.model.eq(ref)].set_index(["season", "round"]).winner
                print(name, "minus", ref, W.paired_boot(a, b), flush=True)
    R.to_csv("results/e175_per_race.csv", index=False)
    P.to_csv("results/e175_predictions.csv", index=False)


if __name__ == "__main__":
    main()
