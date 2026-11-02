import sys; sys.path.insert(0, ".")
from experiments.common import *
X = table(cache=False)
BASE = pre_weekend_cols(X); ALLP = [c for c in PROFILE_COLS(X) if c in X]
has = lambda *pre: [c for c in ALLP if c.startswith(pre)]
B = {
 "base": [],
 "+strategy/attrition": [c for c in ALLP if c in ("c_dnf","c_coll","c_chaos","d_coll","stops_med","stops_2plus","stint_len","soft","hard","wet","sc_frac","vsc_frac")],
 "+fit(car x track)": has("fit_car"),
 "+fit(style x track)": has("fit_style"),
 "+telemetry car profile": has("tp_apex","tp_brake","tp_pick","tp_gain","tp_exitv","tp_acc","tp_vmax","tp_ft","tp_brk"),
 "+driver style": has("ds_"),
 "+pit/exec/deg/sectors": [c for c in ALLP if c.startswith(("tp_pit","tp_exec","tp_deg","tp_s1","tp_s2","tp_s3","tp_st","tp_lr","tp_q","ds_pit","ds_exec"))],
 "+PU(engine) pooled": has("pu_"),
 "+circuit layout": [c for c in ALLP if c.startswith(("c_len","c_slow","c_med","c_fast","c_ft","c_brk","c_vmax","c_vmean","c_longest","c_lowspeed","c_ncor","f_"))],
 "ALL": ALLP,
}
print({k: len(v) for k, v in B.items()})
for nm, extra in B.items():
    F = BASE + extra
    W.MODELS[f"lg|{nm}"] = lambda tr, te, F_, F=F, **kw: W.m_logit_pod(tr, te, F, C=0.03, hl=100)
    W.MODELS[f"ridge|{nm}"] = lambda tr, te, F_, F=F, **kw: reg(tr, te, F, "y_pace", "ridge", alpha=500)
    W.MODELS[f"et|{nm}"] = lambda tr, te, F_, F=F, **kw: reg(tr, te, F, "y_pace", "et")
W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
ms = ["standings"] + [m for m in W.MODELS if "|" in m]
s, R, P = run(X, ms, [2023, 2024, 2025])
s.to_csv("results/e11_blocks_dev.csv")
pd.set_option("display.width", 200)
print(s.sort_index().to_string())
t = s.reset_index(); t[["learner", "block"]] = t["model"].str.split("|", expand=True)
print("\npodium by block x learner:"); print(t.pivot(index="block", columns="learner", values="podium").round(3).to_string())
print("\npace rank-corr by block x learner:"); print(t.pivot(index="block", columns="learner", values="pace_rho").round(3).to_string())
