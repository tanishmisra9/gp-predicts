"""E169: race-conditional, winner-only likelihood on existing pre-weekend features."""
from functools import partial

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import f1.walkforward as W
from experiments.e168_winner import win_score
from f1.data import DATA, load_results
from f1.features import build_features, pre_weekend_cols


def objective(beta, A, groups, winners, weights):
    utility = A @ beta
    loss = beta @ beta / (2 * 0.03)
    grad = beta / 0.03
    for ids, winner, weight in zip(groups, winners, weights):
        u = utility[ids]
        loss += weight * (logsumexp(u) - utility[winner])
        grad += weight * (softmax(u) @ A[ids] - A[winner])
    return loss, grad


def choice_score(tr, te, F, shuffled=False):
    W.assert_pre_weekend(F)
    assert tr.k.max() < te.k.min()
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    A = pipe.fit_transform(tr[F]); B = pipe.transform(te[F])
    groups = list(tr.groupby("k", sort=True).indices.values())
    labels = tr.win.to_numpy()
    assert all(labels[ids].sum() == 1 for ids in groups)
    rng = np.random.default_rng(169)
    winners = [rng.choice(ids) if shuffled else ids[np.argmax(labels[ids])] for ids in groups]
    weights = [W._w(tr, 100).iloc[ids[0]] for ids in groups]
    fitted = minimize(objective, np.zeros(A.shape[1]), args=(A, groups, winners, weights),
                      jac=True, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-11})
    assert fitted.success, fitted.message
    probability = softmax(B @ fitted.x)
    assert np.isfinite(probability).all() and np.isclose(probability.sum(), 1)
    return -probability


def main():
    df = load_results().loc[lambda d: d.season <= 2025]
    pace = pd.read_csv(DATA / "racepace_ff1.csv").loc[lambda d: d.season <= 2025]
    X = build_features(df, pace); F = pre_weekend_cols(X)
    W.assert_pre_weekend(F)
    W.MODELS.update(
        standings=lambda tr, te, F: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values,
        pre_logit=lambda tr, te, F: W.m_logit_pod(tr, te, F, C=0.03, hl=100),
        win_logit=win_score,
        choice_win=choice_score,
        choice_null=partial(choice_score, shuffled=True),
    )
    models = ["standings", "pre_logit", "win_logit", "choice_win", "choice_null"]
    R, P = W.walk_forward(X, list(range(2020, 2026)), models, F=F,
                          ens={"ens": ["standings", "pre_logit"]}, tie=None)
    picked = P.sort_values(["season", "round", "model", "score", "driver"]).drop_duplicates(["season", "round", "model"])
    truth = X.loc[X.win.eq(1), ["season", "round", "driver"]]
    checked = picked.merge(truth, on=["season", "round"], suffixes=("_pred", "_true"))
    independent = checked.assign(hit=checked.driver_pred.eq(checked.driver_true)).groupby("model").hit.sum()
    assert independent.eq(R.groupby("model").winner.sum()).all()
    probs = P[P.model.isin(["win_logit", "pre_logit", "choice_win", "choice_null"])].merge(
        X[["season", "round", "driver", "win"]], on=["season", "round", "driver"], validate="many_to_one")
    probs["prob"] = -probs.score / (-probs.score).groupby([probs.season, probs["round"], probs.model]).transform("sum")
    for split, years in (("inner", [2020, 2021, 2022]), ("outer", [2023, 2024, 2025])):
        d = R[R.season.isin(years)]
        print(split, d.groupby("model")[["winner", "podium"]].agg(["sum", "mean"]).round(4).to_string(), flush=True)
        print(d.pivot_table(index="season", columns="model", values="winner", aggfunc="sum").astype(int).to_string(), flush=True)
        a = d[d.model == "choice_win"].set_index(["season", "round"]).winner
        for ref in ("standings", "ens", "win_logit"):
            b = d[d.model == ref].set_index(["season", "round"]).winner
            print("choice_win minus", ref, W.paired_boot(a, b), flush=True)
        p = probs[probs.season.isin(years) & probs.win.eq(1)]
        print("winner log-loss", p.assign(loss=-np.log(p.prob.clip(lower=1e-15))).groupby("model").loss.mean().to_dict(), flush=True)
    R.to_csv("results/e169_per_race.csv", index=False)
    P.to_csv("results/e169_predictions.csv", index=False)


if __name__ == "__main__":
    main()
