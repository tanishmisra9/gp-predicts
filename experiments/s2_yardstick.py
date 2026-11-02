import sys; sys.path.insert(0, ".")
from experiments.common import *
X = table(cache=False); BASE = pre_weekend_cols(X)
SM = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
W.MODELS["standings"] = SM; W.MODELS["logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, BASE, C=0.03, hl=100)
def rankprior(tr, te, F, **kw):   # calibrated P(podium | standings rank): learned from TRAINING races, returned as -prob
    def rnk(d): return d.groupby("k").d_season_pts.rank(ascending=False, method="first").fillna(20)
    t = tr.assign(r=rnk(tr)); p = t.groupby("r").podium.mean(); return -(rnk(te).map(p).fillna(0.0).values)
W.MODELS["rankprior"] = rankprior
EVAL = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
R, P = W.walk_forward(X, EVAL, ["standings", "logit", "rankprior"], ens={"ens": ["standings", "logit"]}, min_train_season=2018, tie=None)
print(f"=== E63 POOLED evaluation 2019-2025 ({R.groupby(['season','round']).ngroups} races, training from 2018)")
print(W.summarize(R)[["winner", "podium", "ndcg3", "top10", "pairwise", "mae"]].to_string())
print(R.groupby(["model", "season"]).podium.mean().unstack().round(3).to_string())
a = R[R.model == "ens"].set_index(["season", "round"]).podium; b = R[R.model == "standings"].set_index(["season", "round"]).podium
d = (a - b); rng = np.random.default_rng(0); seasons = d.index.get_level_values(0).unique()
bs = [pd.concat([d[d.index.get_level_values(0) == s] for s in rng.choice(seasons, len(seasons))]).mean() for _ in range(4000)]
print(f"=== E68 season-block bootstrap of (ens - standings) podium: mean {d.mean():+.3f}, 95% CI [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]  (per-race paired CI: {np.round(W.paired_boot(a,b)[1],3).tolist()})")
print("\n=== E64 PROBABILISTIC selection metric: Brier & log-loss of P(podium) over all driver-races (lower=better)")
pp = P[P.model.isin(["logit", "rankprior"])].merge(X[["season", "round", "driver", "podium"]], on=["season", "round", "driver"]); pp["p"] = (-pp.score).clip(1e-4, 1 - 1e-4)
pp["brier"] = (pp.p - pp.podium) ** 2; pp["ll"] = -(pp.podium * np.log(pp.p) + (1 - pp.podium) * np.log(1 - pp.p))
pp["sp"] = np.where(pp.season <= 2022, "in 19-22", "out 23-25")
print(pp.groupby(["model", "sp"])[["brier", "ll"]].mean().round(4).unstack().to_string()); print("  constant-0.15 baseline Brier = %.4f" % ((0.15*0.85)))
print("\n=== E65 REVERSED windows: tune logit(C,hl) on OUTER 2023-25, test on INNER 2019-22 (from the 12-setting grid)")
import itertools
rows = []
for C, hl in itertools.product((0.01, 0.03, 0.1, 0.3), (30, 100, 300)):
    W.MODELS[f"lg|{C}|{hl}"] = lambda tr, te, F, C=C, hl=hl, **kw: W.m_logit_pod(tr, te, BASE, C=C, hl=hl)
R2, _ = W.walk_forward(X, EVAL, [m for m in W.MODELS if m.startswith("lg|")] + ["standings"], min_train_season=2018, tie=None)
t = R2.assign(sp=np.where(R2.season <= 2022, "in", "out")).groupby(["model", "sp"]).podium.mean().unstack()
lg = t[t.index.str.startswith("lg|")]; best = lg.out.idxmax()
print(f"  best on OUTER = {best}: outer {lg.loc[best,'out']:.3f} -> INNER {lg.loc[best,'in']:.3f}   | standings inner {t.loc['standings','in']:.3f} outer {t.loc['standings','out']:.3f}")
print(f"  all 12 learned settings: INNER mean {lg['in'].mean():.3f} (range {lg['in'].min():.3f}-{lg['in'].max():.3f}); OUTER mean {lg.out.mean():.3f} (range {lg.out.min():.3f}-{lg.out.max():.3f})")
