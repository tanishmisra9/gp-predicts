import sys, itertools; sys.path.insert(0, ".")
from experiments.common import *
X = table(); BASE = pre_weekend_cols(X); DEV = [2020, 2021, 2022, 2023, 2024, 2025]
M = {}
M["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
for C, hl in itertools.product((0.01, 0.03, 0.1, 0.3), (30, 100, 300)):
    M[f"logit|C{C}|hl{hl}"] = lambda tr, te, F, C=C, hl=hl, **kw: W.m_logit_pod(tr, te, BASE, C=C, hl=hl)
M["lgb_pod"] = lambda tr, te, F, **kw: W.m_lgb_pod(tr, te, BASE, hl=100)
M["lgb_rank"] = lambda tr, te, F, **kw: W.m_lgb_rank(tr, te, BASE, hl=100)
M["ridge_pace"] = lambda tr, te, F, **kw: reg(tr, te, BASE, "y_pace", "ridge", alpha=500)
M["et_pace"] = lambda tr, te, F, **kw: reg(tr, te, BASE, "y_pace", "et")
M["lgb_pace"] = lambda tr, te, F, **kw: reg(tr, te, BASE, "y_pace", "lgb")
M["et_order"] = lambda tr, te, F, **kw: reg(tr, te, BASE, "order", "et")
W.MODELS.update(M)
R, P = W.walk_forward(X, DEV, list(M), tie=None)
P.to_pickle("data/cache_P.pkl")
S = W.summarize(R)[["winner", "podium", "ndcg3", "pairwise", "mae"]]
inner = lambda d: d[d.season <= 2022]; outer = lambda d: d[d.season >= 2023]
print("INNER (2020-22) vs OUTER (2023-25) podium:")
t = R.assign(split=np.where(R.season <= 2022, "inner", "outer")).groupby(["model", "split"]).podium.mean().unstack().round(3)
print(t.sort_values("inner", ascending=False).to_string())
# nested tuning of the logit grid: pick by inner, report outer
lg = t[t.index.str.startswith("logit")]; best = lg.inner.idxmax(); print(f"\nnested-tuned logit = {best}: inner {lg.loc[best,'inner']:.3f} -> OUTER {lg.loc[best,'outer']:.3f}   (hand-set logit|C0.03|hl100 outer {lg.loc['logit|C0.03|hl100','outer']:.3f})")
