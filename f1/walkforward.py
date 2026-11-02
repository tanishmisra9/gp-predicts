"""Walk-forward evaluation: for each target race, fit ONLY on strictly earlier races, predict, score, roll on."""
import warnings
from itertools import combinations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
ID = ["season", "round", "k", "driver", "team", "code", "era", "order", "podium", "win", "season_round"]


def feature_cols(X):
    return [c for c in X if c not in ID]


def metrics(g, score):
    """g: one race's rows; score: lower = predicted to finish better. Ties broken by grid."""
    s = pd.Series(np.asarray(score, float), index=g.index)
    order = s.to_frame("s").join(g.gridn).sort_values(["s", "gridn"]).index
    pred = list(g.loc[order, "driver"]); true = list(g.sort_values("order").driver)
    rk = g.order.values; sc = s.values
    c = n = 0
    for i, j in combinations(range(len(g)), 2):
        if rk[i] != rk[j] and sc[i] != sc[j]:
            n += 1; c += (rk[i] < rk[j]) == (sc[i] < sc[j])
    prank = pd.Series(range(1, len(order) + 1), index=order).reindex(g.index).values
    return dict(winner=float(pred[0] == true[0]), podium=len(set(pred[:3]) & set(true[:3])) / 3,
                exact_podium=float(set(pred[:3]) == set(true[:3])), top5=len(set(pred[:5]) & set(true[:5])) / 5,
                top10=len(set(pred[:10]) & set(true[:10])) / 10, pairwise=c / max(n, 1), mae=np.abs(prank - rk).mean())


# ---- models: each returns a score (lower = better) for the test rows --------------------------------------
def _w(train, hl):  # recency weights over races
    return 0.5 ** ((train.k.max() - train.k) / hl)


def m_grid(tr, te, F, **kw):
    return te.gridn.values


def _lgb_params(**kw):
    return dict(n_estimators=250, learning_rate=0.03, num_leaves=8, min_child_samples=30, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, random_state=0, verbose=-1,
                deterministic=True, force_row_wise=True, n_jobs=1) | kw


def m_lgb_reg(tr, te, F, hl=80, **kw):
    m = lgb.LGBMRegressor(**_lgb_params()).fit(tr[F], tr.order, sample_weight=_w(tr, hl))
    return m.predict(te[F])


def m_lgb_pod(tr, te, F, hl=80, **kw):
    m = lgb.LGBMClassifier(**_lgb_params()).fit(tr[F], tr.podium, sample_weight=_w(tr, hl))
    return -m.predict_proba(te[F])[:, 1]


def m_lgb_rank(tr, te, F, hl=80, **kw):
    tr = tr.sort_values("k")
    rel = np.clip(23 - tr.order.values, 0, None)  # lambdarank gain over top positions
    m = lgb.LGBMRanker(**_lgb_params(objective="lambdarank", lambdarank_truncation_level=10)).fit(
        tr[F], rel, group=tr.groupby("k", sort=True).size().values, sample_weight=None)
    return -m.predict(te[F])


def m_logit_pod(tr, te, F, hl=80, C=0.1, **kw):
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, max_iter=500))
    pipe.fit(tr[F], tr.podium, logisticregression__sample_weight=_w(tr, hl))
    return -pipe.predict_proba(te[F])[:, 1]


MODELS = dict(grid=m_grid, lgb_reg=m_lgb_reg, lgb_pod=m_lgb_pod, lgb_rank=m_lgb_rank, logit_pod=m_logit_pod)


def _rank01(x):  # within-race rank -> [0,1], so heterogeneous model scores can be averaged
    return pd.Series(x).rank(method="average").values / len(x)


def walk_forward(X, seasons, models, F=None, min_train_season=2019, ens=None, **kw):
    """Returns per-race metric table (long): season, round, model, metrics...; ens = {name: [model names]} rank-averaged."""
    F = F or feature_cols(X)
    rows, preds = [], []
    for (s, r), te in X[X.season.isin(seasons)].groupby(["season", "round"], sort=True):
        tr = X[(X.k < te.k.iloc[0]) & (X.season >= min_train_season)]  # strictly earlier races only
        sc = {}
        for name in models:
            sc[name] = MODELS[name](tr, te, F, **kw) if len(tr) else te.gridn.values
        for name, parts in (ens or {}).items():
            sc[name] = np.mean([_rank01(sc[p]) for p in parts], axis=0)
        for name, v in sc.items():
            rows.append(dict(season=s, round=r, model=name, **metrics(te, v)))
            preds.append(te[["season", "round", "driver"]].assign(model=name, score=v))
    return pd.DataFrame(rows), pd.concat(preds)


def summarize(R, by="model"):
    return R.groupby(by)[["winner", "podium", "exact_podium", "top5", "top10", "pairwise", "mae"]].mean().round(3)


def boot_ci(x, n=4000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    b = rng.choice(x, (n, len(x))).mean(1)
    return np.percentile(b, [2.5, 97.5])
