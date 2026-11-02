import sys; sys.path.insert(0, ".")
from experiments.common import *
from sklearn.linear_model import LogisticRegression
X = table(); P = pd.read_pickle("data/cache_P.pkl"); DEV = [2020, 2021, 2022, 2023, 2024, 2025]
DF = [c for c in X if c.split("_", 1)[0] in ("d3", "d10", "ds") and c not in ("ds_coll",)] + ["d_season_pts", "prev_order", "d_coll", "d_circ_res"]
DF = [c for c in dict.fromkeys(DF) if c in X]
A = X[["k", "season", "round", "team", "driver", "order"] + DF]
pairs = A.merge(A, on=["k", "season", "round", "team"], suffixes=("", "_b")); pairs = pairs[pairs.driver != pairs.driver_b]
for c in DF: pairs["x_" + c] = pairs[c] - pairs[c + "_b"]
pairs["y"] = (pairs.order < pairs.order_b).astype(int); XF = ["x_" + c for c in DF]
out = []
for k in sorted(pairs[pairs.season.isin(DEV)].k.unique()):
    tr = pairs[(pairs.k < k) & (pairs.season >= 2019)]; te = pairs[pairs.k == k]
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.05, max_iter=500)).fit(tr[XF], tr.y, logisticregression__sample_weight=W._w(tr, 100))
    out.append(te[["season", "round", "driver", "driver_b", "y"]].assign(p=m.predict_proba(te[XF])[:, 1]))
T = pd.concat(out)
print("teammate head-to-head accuracy of dedicated model, 2020-25: %.3f (n=%d pairs)   [ens-implied was 0.639 on classified pairs; coin flip 0.5]" % (((T.p > .5) == (T.y == 1)).mean(), len(T) // 2))
print("   inner 2020-22: %.3f   outer 2023-25: %.3f" % tuple(((T[T.season.between(a, b)].p > .5) == (T[T.season.between(a, b)].y == 1)).mean() for a, b in ((2020, 2022), (2023, 2025))))
# combine: ens score + lam * (P(ahead of teammate) - 0.5)
pw = P[P.model.isin(["standings", "logit|C0.03|hl100"])].copy(); pw["r"] = pw.groupby(["model", "season", "round"]).score.rank(pct=True)
ens = pw.groupby(["season", "round", "driver"]).r.mean().rename("ens").reset_index()
pm = T.groupby(["season", "round", "driver"]).p.mean().rename("pt").reset_index()
D = X[["season", "round", "driver", "order", "gridn"]].merge(ens, on=["season", "round", "driver"]).merge(pm, on=["season", "round", "driver"], how="left"); D["pt"] = D.pt.fillna(.5)
def ev(lam, seasons):
    o = [W.metrics(g, g.ens - lam * (g.pt - .5), tie=None) for _, g in D[D.season.isin(seasons)].groupby(["season", "round"])]
    d = pd.DataFrame(o); return d.podium.mean(), d.ndcg3.mean()
print("\nlambda  inner-podium inner-ndcg | outer-podium outer-ndcg")
for lam in (0, .1, .2, .4, .8):
    a, b = ev(lam, [2020, 2021, 2022]), ev(lam, [2023, 2024, 2025]); print(f"{lam:5.1f}   {a[0]:.3f}  {a[1]:.3f}   |   {b[0]:.3f}  {b[1]:.3f}")
