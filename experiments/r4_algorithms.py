import sys; sys.path.insert(0, ".")
from experiments.common import *
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
X = table(); BASE = pre_weekend_cols(X); DEV = [2020, 2021, 2022, 2023, 2024, 2025]
W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
W.MODELS["logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, BASE, C=0.03, hl=100)
def sk(model, scale=True, sw=True, F=None):
    def f(tr, te, _F, **kw):
        cols = F or BASE; steps = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else []) + [model()]
        p = make_pipeline(*steps); name = p.steps[-1][0]
        try: p.fit(tr[cols], tr.podium, **({f"{name}__sample_weight": W._w(tr, 100)} if sw else {}))
        except TypeError: p.fit(tr[cols], tr.podium)
        return -p.predict_proba(te[cols])[:, 1]
    return f
KEY = ["d3_order", "d10_order", "t3_order", "t10_order", "d_season_pts", "t_season_pts", "d10_pts_n", "t10_pts_n", "d3_q_pos", "t3_q_pos"]
KEY = [c for c in KEY if c in X]
W.MODELS.update({
 "E47_svm_rbf": sk(lambda: SVC(C=0.3, gamma="scale", probability=True, random_state=0), F=KEY, sw=False),
 "E48_knn_k60": sk(lambda: KNeighborsClassifier(60, weights="distance"), F=KEY, sw=False),
 "E49_naive_bayes": sk(lambda: GaussianNB(), F=KEY, sw=False),
 "E50_adaboost": sk(lambda: AdaBoostClassifier(n_estimators=150, learning_rate=0.3, random_state=0), scale=False),
 "E51_hist_gb": sk(lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=5.0, random_state=0), scale=False),
 "E52_random_forest": sk(lambda: RandomForestClassifier(400, min_samples_leaf=15, max_features=0.4, random_state=0, n_jobs=4), scale=False),
 "E53_bagged_logit": sk(lambda: BaggingClassifier(LogisticRegression(C=0.03, max_iter=300), n_estimators=25, max_samples=0.7, random_state=0), sw=False),
})
def mono(tr, te, F, **kw):   # monotone-constrained GBM: better form (lower order/gap) can never lower podium probability
    cols = [c for c in BASE]; cons = [(-1 if any(s in c for s in ("order", "q_pos", "q_gap", "pace_gap", "dnf")) else (1 if ("pts" in c) else 0)) for c in cols]
    m = lgb.LGBMClassifier(**W._lgb_params(monotone_constraints=cons, monotone_constraints_method="advanced")).fit(tr[cols], tr.podium, sample_weight=W._w(tr, 100)); return -m.predict_proba(te[cols])[:, 1]
W.MODELS["E54_monotone_gbm"] = mono
def top8(tr, te, F, **kw):   # features chosen from TRAINING races only by |corr with podium|
    c = tr[BASE].apply(lambda s: s.fillna(s.median())).corrwith(tr.podium).abs().sort_values(ascending=False).head(8).index.tolist()
    return W.m_logit_pod(tr, te, c, C=0.1, hl=100)
W.MODELS["E55_top8_features_logit"] = top8
ms = ["standings", "logit"] + [m for m in W.MODELS if m.startswith("E")]
R, P = W.walk_forward(X, DEV, ms, ens={"ens": ["standings", "logit"]}, tie=None); report(R, "ens"); P.to_pickle("data/cache_P_r4.pkl")
