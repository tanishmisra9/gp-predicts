import sys; sys.path.insert(0, ".")
from experiments.common import *
X = table(cache=False); BASE = pre_weekend_cols(X)
DEV = [2020, 2021, 2022, 2023, 2024, 2025]
W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
W.MODELS["form_d3"] = lambda tr, te, F, **kw: te.d3_order.fillna(22).values
W.MODELS["pre_logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, BASE, C=0.03, hl=100)
def shuffled(tr, te, F, **kw):   # NULL TEST: train on label-shuffled data (podium labels permuted across driver-rows) -> must collapse to ~chance
    tr = tr.copy(); tr["podium"] = np.random.default_rng(0).permutation(tr.podium.values); return W.m_logit_pod(tr, te, BASE, C=0.03, hl=100)
W.MODELS["NULL_shuffled_labels"] = shuffled
W.MODELS["random"] = lambda tr, te, F, **kw: np.random.default_rng(len(te) + int(te.k.iloc[0])).random(len(te))
s, R, P = run(X, ["standings", "form_d3", "pre_logit", "NULL_shuffled_labels", "random"], DEV, ens={"ens": ["standings", "pre_logit"]})
print(f"{R.groupby(['season','round']).ngroups} races, dev 2020-2025, tie-break = driver name")
print(s.to_string())
e = R[R.model == "ens"].set_index(["season", "round"]).podium; st = R[R.model == "standings"].set_index(["season", "round"]).podium
print("ens vs standings podium, paired:", [np.round(x, 3) if not isinstance(x, np.ndarray) else x.round(3).tolist() for x in W.paired_boot(e, st)])
print(R.groupby(["model", "season"]).podium.mean().unstack().round(3).loc[["ens", "standings", "pre_logit", "random"]].to_string())
