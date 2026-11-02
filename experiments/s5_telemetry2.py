import sys, glob; sys.path.insert(0, ".")
from experiments.common import *
from f1.profile import load_aux, _driver_weekend, TEL, PROFILE_COLS, add_profile
X = table(); BASE = pre_weekend_cols(X); res = load_results(); df = res
aux = list(load_aux()); tel1 = aux[0]
tel2 = pd.concat([pd.read_csv(f) for f in glob.glob("data/parts/tel2_*.csv")]).drop_duplicates(["season", "round", "code"])
print("=== E83 RELIABILITY AUDIT v1 (1 fastest lap) vs v2 (mean of 3 fastest laps), 2022-26 weekends")
pace = aux[5]
def audit(tel):
    w, M = _driver_weekend(df, tel, aux[3], aux[4], pace); w = w[w.season >= 2022]; rows = []
    for c in [c for c in TEL if c in w and w[c].notna().mean() > .5]:
        d = w.dropna(subset=["t_" + c]); t = d.groupby(["season", "round", "team"])["t_" + c].first().reset_index().sort_values(["team", "season", "round"]); t["prev"] = t.groupby(["team", "season"])["t_" + c].shift(1); t = t.dropna()
        wk = np.corrcoef(t["t_" + c], t.prev)[0, 1]; icc = d.groupby(["season", "team"])[c].mean().var() / d[c].var()
        m = d.merge(pace, on=["season", "round", "code"]); val = m.groupby(["season", "round"]).apply(lambda x: x[c].corr(x.pace_gap, method="spearman"), include_groups=False).mean(); rows.append((c, wk, icc, val))
    return pd.DataFrame(rows, columns=["metric", "persist", "between", "valid"]).set_index("metric")
a1, a2 = audit(tel1), audit(tel2); cmp = a1.join(a2, lsuffix="_v1", rsuffix="_v2"); print(cmp.round(2).to_string())
print("  MEDIAN  v1: persistence %.2f between-team share %.2f validity %.2f | v2: persistence %.2f between-team share %.2f validity %.2f" % (*a1.median(), *a2.median()))
aux2 = list(aux); aux2[0] = tel2
Xp = add_profile(X.drop(columns=PROFILE_COLS(X), errors="ignore"), df, tuple(aux2))   # drop v1 profile columns first (else merge suffixes collide)
Xp = Xp.merge(pd.concat([pd.read_csv(f) for f in glob.glob("data/parts/tel2_*.csv")])[["season", "round"]].drop_duplicates().assign(_t=1), on=["season", "round"], how="left")
PROF = [c for c in PROFILE_COLS(Xp) if c.startswith(("tp_", "ds_", "fit_"))]; keep = list(a2[(a2.persist > .15)].index); KEEP = [f"tp_{m}" for m in keep if f"tp_{m}" in Xp] + [f"fit_car_{m.split('_')[0]}" for m in keep if f"fit_car_{m.split('_')[0]}" in Xp]
print(f"\n=== E85 persistent-only metrics (v2 persistence>0.15): {keep}")
DEV = [2022, 2023, 2024, 2025]
W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
for nm, ex in {"base": [], "E84_v2_full_profile": PROF, "E85_v2_persistent_only": KEEP}.items():
    F = BASE + ex
    W.MODELS[f"lg|{nm}"] = lambda tr, te, F_, F=F, **kw: W.m_logit_pod(tr, te, F, C=0.03, hl=100)
    W.MODELS[f"ridge|{nm}"] = lambda tr, te, F_, F=F, **kw: reg(tr, te, F, "y_pace", "ridge", alpha=500)
    W.MODELS[f"et|{nm}"] = lambda tr, te, F_, F=F, **kw: reg(tr, te, F, "y_pace", "et")
s, R, P = run(Xp, ["standings"] + [m for m in W.MODELS if "|" in m], DEV)
print("\n=== E84/E85 pre-weekend, 2022-25 (92 races): podium / ndcg3 / pace-order rho"); print(s.sort_index().to_string())
