"""Shared harness for pre-weekend experiments: feature table, extra regression/learner types, and the low-noise pace-rank diagnostic."""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import f1.walkforward as W
from f1.data import DATA, load_results
from f1.features import build_features, pre_weekend_cols
from f1.profile import PROFILE_COLS, add_profile

warnings.filterwarnings("ignore")


def table(cache=True):
    f = DATA / "cache_X.pkl"
    if cache and f.exists():
        return pd.read_pickle(f)
    df = load_results(); pace = pd.read_csv(DATA / "racepace_ff1.csv")
    X = add_profile(build_features(df, pace), df)
    # TRAINING LABELS ONLY (target-race pace / quali gap): used exclusively from strictly earlier races by walk_forward
    X = X.merge(pace.rename(columns={"pace_gap": "y_pace"}), on=["season", "round", "code"], how="left")
    X["y_q"] = X["q_gap"]
    X.to_pickle(f); return X


def reg(tr, te, F, y, kind="lgb", hl=100, **kw):
    t = tr[tr[y].notna()]; w = W._w(t, hl)
    if kind == "lgb":
        m = lgb.LGBMRegressor(**W._lgb_params(n_estimators=300, learning_rate=0.03)).fit(t[F], t[y], sample_weight=w); return m.predict(te[F])
    if kind == "et":
        p = make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(300, min_samples_leaf=8, max_features=0.5, random_state=0, n_jobs=4))
        p.fit(t[F], t[y], extratreesregressor__sample_weight=w); return p.predict(te[F])
    if kind == "ridge":
        p = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=kw.get("alpha", 300)))
        p.fit(t[F], t[y], ridge__sample_weight=w); return p.predict(te[F])
    if kind == "mlp":
        p = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), MLPRegressor((32, 16), alpha=1.0, max_iter=400, early_stopping=True, random_state=0))
        p.fit(t[F], t[y]); return p.predict(te[F])


def pace_corr(P, X):
    """mean per-race Spearman between predicted order and TRUE race-pace order (higher=better; diagnostic, low noise)."""
    d = P.merge(X[["season", "round", "driver", "y_pace"]], on=["season", "round", "driver"]).dropna(subset=["y_pace"])
    return d.groupby(["model", "season", "round"]).apply(lambda g: spearmanr(g.score, g.y_pace)[0], include_groups=False).groupby("model").mean().round(3)


def run(X, models, seasons, ens=None):
    R, P = W.walk_forward(X, seasons, models, ens=ens, tie=None)  # tie=None: never break ties with the target-weekend grid
    s = W.summarize(R)[["winner", "podium", "ndcg3", "top10", "pairwise", "mae"]]
    s["pace_rho"] = pace_corr(P, X)
    return s.sort_values("podium", ascending=False), R, P


def report(R, ref="ens", title=""):
    """inner(2020-22)/outer(2023-25) podium & ndcg3 per model + paired outer diff vs ref (races)."""
    R = R.assign(sp=np.where(R.season <= 2022, "in", "out"))
    t = R.groupby(["model", "sp"])[["podium", "ndcg3"]].mean().unstack().round(3); t.columns = [f"{a}_{b}" for a, b in t.columns]
    if ref in set(R.model):
        a = R[(R.model == ref) & (R.sp == "out")].set_index(["season", "round"]).podium
        t["dOuter_vs_" + ref] = [round(W.paired_boot(R[(R.model == m) & (R.sp == "out")].set_index(["season", "round"]).podium, a)[0], 3) for m in t.index]
    print(t.sort_values("podium_out", ascending=False).to_string())
