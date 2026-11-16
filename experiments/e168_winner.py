"""E168: one fixed pre-weekend winner-label classifier versus standings and frozen ens."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import f1.walkforward as W
from f1.data import DATA, load_results
from f1.features import build_features, pre_weekend_cols


def win_score(tr, te, F, shuffled=False):
    y = tr.win.to_numpy().copy()
    if shuffled:
        y[:] = 0
        rng = np.random.default_rng(168)
        for indices in tr.groupby("k", sort=True).indices.values():
            y[rng.choice(indices)] = 1
        assert y.sum() == tr.k.nunique()
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.03, max_iter=500))
    model.fit(tr[F], y, logisticregression__sample_weight=W._w(tr, 100))
    return -model.predict_proba(te[F])[:, 1]


def main():
    df = load_results().loc[lambda d: d.season <= 2025]
    pace = pd.read_csv(DATA / "racepace_ff1.csv").loc[lambda d: d.season <= 2025]
    X = build_features(df, pace)
    F = pre_weekend_cols(X)
    W.assert_pre_weekend(F)
    W.MODELS.update(
        standings=lambda tr, te, F: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values,
        pre_logit=lambda tr, te, F: W.m_logit_pod(tr, te, F, C=0.03, hl=100),
        win_logit=lambda tr, te, F: win_score(tr, te, F),
        win_null=lambda tr, te, F: win_score(tr, te, F, shuffled=True),
    )
    R, P = W.walk_forward(X, list(range(2020, 2026)), ["standings", "pre_logit", "win_logit", "win_null"], F=F,
                          ens={"ens": ["standings", "pre_logit"]}, tie=None)
    picked = P.sort_values(["season", "round", "model", "score", "driver"]).drop_duplicates(["season", "round", "model"])
    actual = X.sort_values("order").drop_duplicates(["season", "round"])[["season", "round", "driver"]]
    checked = picked.merge(actual, on=["season", "round"], suffixes=("_pred", "_true"))
    independent = checked.assign(hit=checked.driver_pred.eq(checked.driver_true)).groupby("model").hit.sum()
    assert independent.eq(R.groupby("model").winner.sum()).all()
    assert R.season.max() == 2025
    for name, years in (("inner", [2020, 2021, 2022]), ("outer", [2023, 2024, 2025])):
        d = R[R.season.isin(years)]
        print(name, "races", d[d.model == "standings"].shape[0])
        print(d.groupby("model")[["winner", "podium"]].agg(["sum", "mean"]).round(4).to_string())
        print("yearly winner counts")
        print(d.pivot_table(index="season", columns="model", values="winner", aggfunc="sum").astype(int).to_string())
        a = d[d.model == "win_logit"].set_index(["season", "round"]).winner
        for ref in ("standings", "ens"):
            b = d[d.model == ref].set_index(["season", "round"]).winner
            diff, ci, p = W.paired_boot(a, b)
            print(f"win_logit - {ref}: {diff:+.4f} CI [{ci[0]:+.4f}, {ci[1]:+.4f}] P(diff<=0)={p:.4f}")
    assert R.groupby(["season", "round"]).size().eq(5).all()


if __name__ == "__main__":
    main()
