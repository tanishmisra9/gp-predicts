import sys; sys.path.insert(0, ".")
from experiments.common import *
X = table(cache=False)
BASE = pre_weekend_cols(X); PROF = [c for c in PROFILE_COLS(X) if c in X]
print(len(BASE), "base,", len(PROF), "profile features; profile coverage (non-null) by season:")
print(X.groupby("season")[["tp_apex_slow", "ds_apex_slow", "fit_car_apex", "stops_med", "c_dnf", "tp_pit", "tp_exec"]].apply(lambda d: d.notna().mean()).round(2).to_string())
S = {"base": BASE, "base+prof": BASE + PROF}
for nm, F in S.items():
    W.MODELS[f"lg_{nm}"] = lambda tr, te, F_, F=F, **kw: W.m_logit_pod(tr, te, F, C=0.03, hl=100)
    W.MODELS[f"gb_{nm}"] = lambda tr, te, F_, F=F, **kw: W.m_lgb_pod(tr, te, F, hl=100)
    W.MODELS[f"pace_lgb_{nm}"] = lambda tr, te, F_, F=F, **kw: reg(tr, te, F, "y_pace", "lgb")
    W.MODELS[f"pace_ridge_{nm}"] = lambda tr, te, F_, F=F, **kw: reg(tr, te, F, "y_pace", "ridge")
    W.MODELS[f"pace_et_{nm}"] = lambda tr, te, F_, F=F, **kw: reg(tr, te, F, "y_pace", "et")
W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
s, R, P = run(X, ["standings"] + [m for m in W.MODELS if m.startswith(("lg_", "gb_", "pace_"))], [2023, 2024, 2025])
print(s.to_string())
print(R.groupby(["model", "season"]).podium.mean().unstack().round(3).loc[s.index].to_string())
