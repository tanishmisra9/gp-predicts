import sys; sys.path.insert(0, ".")
from experiments.common import *
from scipy.stats import spearmanr
X = table(); BASE = pre_weekend_cols(X)
W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
W.MODELS["pre_logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, BASE, C=0.03, hl=100)
R, P = W.walk_forward(X, [2020, 2021, 2022, 2023, 2024, 2025], ["standings", "pre_logit"], ens={"ens": ["standings", "pre_logit"]}, tie=None)
p = P[P.model == "ens"].merge(X[["season", "round", "driver", "team", "order", "podium", "y_pace", "gridn"]], on=["season", "round", "driver"])
df = load_results(); st = df[["season", "round", "driver", "status"]]
p = p.merge(st, on=["season", "round", "driver"]); p["fin"] = p.status.isin(["Finished", "Lapped"]) | p.status.str.startswith("+")
# 1) teammate order accuracy (both classified)
a = p[p.fin].merge(p[p.fin], on=["season", "round", "team"], suffixes=("", "_b")); a = a[a.driver < a.driver_b]
print("teammate head-to-head (both classified): predicted-better actually finished ahead = %.3f (n=%d)" % (((a.score < a.score_b) == (a.order < a.order_b)).mean(), len(a)))
# 2) team order accuracy
t = p[p.fin].groupby(["season", "round", "team"]).agg(s=("score", "mean"), o=("order", "mean")).reset_index()
print("team-level order Spearman (predicted vs actual, classified cars): %.3f" % t.groupby(["season", "round"]).apply(lambda g: spearmanr(g.s, g.o)[0], include_groups=False).mean())
# 3) classify each MISSED true podium driver and each FALSE predicted podium driver
rows = []
for (s, r), g in p.groupby(["season", "round"]):
    g = g.sort_values("score"); pred = set(g.head(3).driver); true = set(g[g.podium == 1].driver)
    for d in true - pred:
        x = g[g.driver == d].iloc[0]; rows.append(("MISSED real podium driver", "his predicted rank", int(g.driver.tolist().index(d)) + 1))
    for d in pred - true:
        x = g[g.driver == d].iloc[0]; rows.append(("FALSE predicted podium driver", "retired" if not x.fin else ("finished P4-6" if x.order <= 6 else "finished P7+"), 0))
R2 = pd.DataFrame(rows, columns=["kind", "what", "rank"])
f = R2[R2.kind.str.startswith("FALSE")].what.value_counts(normalize=True).round(3)
print("\nFALSE predicted-podium drivers (%d): outcome" % (R2.kind.str.startswith("FALSE")).sum()); print(f.to_string())
m = R2[R2.kind.str.startswith("MISSED")]
print("\nMISSED real-podium drivers (%d): our predicted rank of them -> %s" % (len(m), m["rank"].clip(upper=10).value_counts().sort_index().to_dict()))
# 4) share of all podium slots lost to DNF of predicted drivers
print("\nshare of predicted-podium slots lost to retirement: %.3f   (podium accuracy ceiling if no retirements ~ +%.3f)" % ((R2.what == "retired").sum() / (len(p.groupby(['season', 'round'])) * 3), (R2.what == "retired").sum() / (len(p.groupby(['season', 'round'])) * 3)))
