"""E170: team-win probability times driver-given-winning-team probability."""
from functools import partial

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import f1.walkforward as W
from f1.data import DATA, load_results
from f1.features import build_features, pre_weekend_cols


def fit_probability(tr, te, F):
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.03, max_iter=500))
    model.fit(tr[F], tr.win, logisticregression__sample_weight=W._w(tr, 100))
    return model.predict_proba(te[F])[:, 1]


def team_winner(tr, te, F, shuffled=False):
    W.assert_pre_weekend(F)
    assert tr.k.max() < te.k.min()
    tr = tr.copy(); te = te.copy()
    if shuffled:
        y = np.zeros(len(tr), dtype=int); rng = np.random.default_rng(170)
        for ids in tr.groupby("k", sort=True).indices.values():
            y[rng.choice(ids)] = 1
        tr["win"] = y
    tf = [c for c in F if c.startswith(("t3_", "t10_")) or c in ("t_season_pts", "t_circ_res")]
    df = [c for c in F if c.startswith(("d3_", "d10_")) or c in ("d_season_pts", "prev_order", "d_circ_res")]
    teams = tr.groupby(["k", "team"], sort=True).agg({**{c: "mean" for c in tf}, "win": "max"}).reset_index()
    test_teams = te.groupby("team", sort=True)[tf].mean().reset_index()
    pt = fit_probability(teams, test_teams, tf)
    team_prob = pd.Series(pt / pt.sum(), index=test_teams.team)
    tr[df] = tr[df] - tr.groupby(["k", "team"])[df].transform("mean")
    te[df] = te[df] - te.groupby("team")[df].transform("mean")
    eligible = tr.groupby(["k", "team"]).win.transform("max").eq(1)
    pdv = pd.Series(fit_probability(tr[eligible], te, df), index=te.index)
    pdv = pdv / pdv.groupby(te.team).transform("sum")
    joint = te.team.map(team_prob) * pdv
    assert np.isfinite(joint).all() and np.isclose(joint.sum(), 1)
    return -joint.to_numpy()


def main():
    df = load_results().loc[lambda d: d.season <= 2025]
    pace = pd.read_csv(DATA / "racepace_ff1.csv").loc[lambda d: d.season <= 2025]
    X = build_features(df, pace); F = pre_weekend_cols(X)
    W.MODELS.update(team_win=team_winner, team_null=partial(team_winner, shuffled=True))
    R, P = W.walk_forward(X, list(range(2020, 2026)), ["team_win", "team_null"], F=F, tie=None)
    reference = pd.read_csv("results/e169_per_race.csv")
    ref_pred = pd.read_csv("results/e169_predictions.csv", float_precision="round_trip")
    for frame in (reference, ref_pred):
        assert set(map(tuple, frame[["season", "round"]].to_numpy())) == set(map(tuple, R[["season", "round"]].to_numpy()))
    R = pd.concat([R, reference[reference.model.isin(["standings", "ens", "win_logit", "choice_win"])]])
    P = pd.concat([P, ref_pred[ref_pred.model.isin(["standings", "ens", "win_logit", "choice_win"])]])
    picked = P.sort_values(["score", "driver"]).drop_duplicates(["season", "round", "model"])
    picked = picked.merge(X[["season", "round", "driver", "team"]], on=["season", "round", "driver"], validate="many_to_one")
    truth = X.loc[X.win.eq(1), ["season", "round", "driver", "team"]]
    checked = picked.merge(truth, on=["season", "round"], suffixes=("_pred", "_true"))
    checked["hit"] = checked.driver_pred.eq(checked.driver_true)
    checked["team_hit"] = checked.team_pred.eq(checked.team_true)
    assert checked.groupby("model").hit.sum().eq(R.groupby("model").winner.sum()).all()
    for split, years in (("inner", [2020, 2021, 2022]), ("outer", [2023, 2024, 2025])):
        d = R[R.season.isin(years)]
        print(split, d.groupby("model")[["winner", "podium"]].agg(["sum", "mean"]).round(4).to_string(), flush=True)
        print(d.pivot_table(index="season", columns="model", values="winner", aggfunc="sum").astype(int).to_string(), flush=True)
        print("winning-team hits", checked[checked.season.isin(years)].groupby("model").team_hit.sum().to_dict(), flush=True)
        a = d[d.model.eq("team_win")].set_index(["season", "round"]).winner
        for ref in ("standings", "ens", "choice_win"):
            b = d[d.model.eq(ref)].set_index(["season", "round"]).winner
            print("team_win minus", ref, W.paired_boot(a, b), flush=True)
    R.to_csv("results/e170_per_race.csv", index=False)
    P.to_csv("results/e170_predictions.csv", index=False)


if __name__ == "__main__":
    main()
