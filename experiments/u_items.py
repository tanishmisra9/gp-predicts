import sys, glob, itertools; sys.path.insert(0, ".")
from experiments.common import *
from scipy.stats import spearmanr
X = table(); BASE = pre_weekend_cols(X); res = load_results(); pace = pd.read_csv(DATA / "racepace_ff1.csv")
# ======================= ITEM 2: energy signature audit
en = pd.concat([pd.read_csv(f) for f in glob.glob("data/parts/energy_*.csv")]).drop_duplicates(["season", "round", "code"])
E = en.merge(res[["season", "round", "code", "team", "circuit"]], on=["season", "round", "code"]).merge(pace, on=["season", "round", "code"], how="left")
print("=== ITEM 2: energy-deployment signature (team means of each driver's fastest quali lap)")
rows = []
for regime, (a, b) in {"2022-25 cars": (2022, 2025), "2026 cars": (2026, 2026)}.items():
    for m in ("acc_mid", "acc_hi", "taper", "clip_share"):
        d = E[E.season.between(a, b)].dropna(subset=[m]); t = d.groupby(["season", "round", "team"])[m].mean().reset_index()
        t["c"] = t[m] - t.groupby(["season", "round"])[m].transform("mean"); t = t.sort_values(["team", "season", "round"]); t["prev"] = t.groupby(["team", "season"]).c.shift(1); p = t.dropna(subset=["prev"])
        mm = d.groupby(["season", "round", "team"]).agg(v=(m, "mean"), pace=("pace_gap", "mean")).reset_index(); val = mm.groupby(["season", "round"]).apply(lambda g: spearmanr(g.v, g.pace)[0] if g.pace.notna().sum() > 5 else np.nan, include_groups=False).mean()
        rows.append((regime, m, p.c.corr(p.prev), d.groupby(["season", "team"])[m].mean().var() / d[m].var(), val))
print(pd.DataFrame(rows, columns=["regime", "metric", "week-to-week persistence", "between-team share", "corr w/ race pace (within weekend)"]).round(2).to_string(index=False))
# ======================= ITEM 1b: static circuit descriptors as features (altitude, climate history, tyre hardness history)
cs = pd.read_csv(DATA / "circuit_static.csv")[["circuit", "altitude"]]; wx = pd.read_csv(DATA / "weather.csv")
rc = res.drop_duplicates(["season", "round"])[["season", "round", "circuit"]].merge(X.drop_duplicates(["season", "round"])[["season", "round", "k"]], on=["season", "round"]).merge(wx, on=["season", "round"], how="left").merge(cs, on="circuit", how="left")
import glob as g2
st = pd.concat([pd.read_csv(f) for f in g2.glob("data/parts/strat_*.csv")]); rc = rc.merge(st[["season", "round", "hard", "soft"]], on=["season", "round"], how="left")
for c in ("temp_hist", "rain_hist", "hard_hist"): rc[c] = np.nan
for i, r in rc.iterrows():
    p = rc[(rc.circuit == r.circuit) & (rc.k < r.k)]
    if len(p): rc.loc[i, ["temp_hist", "rain_hist", "hard_hist"]] = [p.temp_act.mean(), p.rain_act.mean(), p.hard.mean()]
X = X.merge(rc[["season", "round", "altitude", "temp_hist", "rain_hist", "hard_hist"]], on=["season", "round"], how="left")
X["alt_hi"] = (X.altitude > 600).astype(float)
for c in ("t10_order", "d10_order", "t10_pace_gap"): X["alt_x_" + c] = X.alt_hi * X[c].fillna(X[c].median()); X["temp_x_" + c] = (X.temp_hist.fillna(24) - 24) / 8 * X[c].fillna(X[c].median())
sd = ["altitude", "alt_hi", "temp_hist", "rain_hist", "hard_hist"] + [c for c in X if c.startswith(("alt_x_", "temp_x_"))]
W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
W.MODELS["logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, BASE, C=0.03, hl=100)
W.MODELS["I1_static_descriptors_logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, BASE + sd, C=0.03, hl=100)
W.MODELS["I1_static_descriptors_lgb"] = lambda tr, te, F, **kw: W.m_lgb_pod(tr, te, BASE + sd, hl=100)
W.MODELS["I1_static_descriptors_ET_pace"] = lambda tr, te, F, **kw: reg(tr, te, BASE + sd, "y_pace", "et")
W.MODELS["lgb_base"] = lambda tr, te, F, **kw: W.m_lgb_pod(tr, te, BASE, hl=100)
W.MODELS["ET_pace_base"] = lambda tr, te, F, **kw: reg(tr, te, BASE, "y_pace", "et")
# ======================= ITEM 3: reliability-weighted latent team strength from FIVE weak indicators (multi-observation Kalman)
q1 = res.assign(q1gap=lambda d: (d.q1 / d.groupby(["season", "round"]).q1.transform("min") - 1) * 100)[["season", "round", "code", "q1gap", "points"]]
ses = pd.concat([pd.read_csv(f) for f in glob.glob("data/parts/sessions_*.csv")]); fp = ses[ses.session.isin(["FP1", "FP2", "FP3"])].groupby(["season", "round", "code"]).lr_gap.mean().rename("fp_lr").reset_index()
pv = pd.concat([pd.read_csv(f) for f in glob.glob("data/parts/pacev_*.csv")]).drop_duplicates(["season", "round", "code"])[["season", "round", "code", "fcorr_gap"]]
A = X[["season", "round", "k", "code", "team", "y_pace", "order"]].merge(q1, on=["season", "round", "code"], how="left").merge(fp, on=["season", "round", "code"], how="left").merge(pv, on=["season", "round", "code"], how="left")
A["ptsneg"] = -A.points; A["ordneg"] = A.order
ind = {"race_pace": "y_pace", "race_pace_fuelcorr": "fcorr_gap", "quali_q1": "q1gap", "fp_longrun": "fp_lr", "points": "ptsneg", "finish_pos": "ordneg"}
T = {}
for nm, c in ind.items():
    t = A.groupby(["k", "season", "team"])[c].mean().reset_index(); t[c] = t[c] - t.groupby("k")[c].transform("mean"); t[c] /= t[c].std(); T[nm] = t.set_index(["k", "team"])[c].to_dict()
teams_by_k = X.groupby("k").team.unique().to_dict(); season_of = X.drop_duplicates("k").set_index("k").season.to_dict()
def fuse(noise, q=0.03, qs=0.4):
    st, out, prev = {}, [], None
    for k in sorted(teams_by_k):
        s = season_of[k]
        for tm in teams_by_k[k]:
            m, P = st.get(tm, [0.0, 3.0]); P += q + (qs if prev is not None and s != prev else 0); st[tm] = [m, P]; out.append((k, tm, m))
        for tm in teams_by_k[k]:
            for nm, r in noise.items():
                if r is None or (k, tm) not in T[nm]: continue
                m, P = st[tm]; y = T[nm][(k, tm)]
                if np.isfinite(y): K = P / (P + r); st[tm] = [m + K * (y - m), (1 - K) * P]
        prev = s
    return pd.DataFrame(out, columns=["k", "team", "f"])
act = A.groupby(["k", "team"]).order.min().reset_index().rename(columns={"order": "bestfin"}); act["season"] = act.k.map(season_of)
def score(f, seasons):
    m = act.merge(f, on=["k", "team"]); m = m[m.season.isin(seasons)]
    return m.groupby("k").apply(lambda g: spearmanr(g.f, g.bestfin)[0] if len(g) > 4 else np.nan, include_groups=False).mean()
inner, outer = [2020, 2021, 2022], [2023, 2024, 2025]
base_noise = {"race_pace": 1.0, "race_pace_fuelcorr": None, "quali_q1": None, "fp_longrun": None, "points": None, "finish_pos": None}
print("\n=== ITEM 3: fused latent team strength -> forecast of TEAM finishing order (Spearman vs actual best-car order)")
print(f"  single indicator (race pace only)      : inner {score(fuse(base_noise), inner):.4f}  outer {score(fuse(base_noise), outer):.4f}")
best = None
grid = [None, 1.0, 3.0, 8.0]
for combo in itertools.product(grid, repeat=len(ind)):
    if all(c is None for c in combo): continue
    noise = dict(zip(ind, combo)); s = score(fuse(noise), inner)
    if best is None or s > best[0]: best = (s, noise)
print(f"  FUSED (noise levels tuned on INNER only): inner {best[0]:.4f}  OUTER {score(fuse(best[1]), outer):.4f}   noise={ {k: v for k, v in best[1].items() if v is not None} }")
allw = {k: 2.0 for k in ind}; print(f"  FUSED equal weights (all 6, r=2)         : inner {score(fuse(allw), inner):.4f}  outer {score(fuse(allw), outer):.4f}")
fz = fuse(best[1]).rename(columns={"f": "fused"}); X = X.merge(fz, on=["k", "team"], how="left")
W.MODELS["I3_fused_strength_only"] = lambda tr, te, F, **kw: te.fused.fillna(te.fused.max()).values
W.MODELS["I3_fused_plus_logit_features"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, BASE + ["fused"], C=0.03, hl=100)
ms = ["standings", "logit", "lgb_base", "ET_pace_base"] + [m for m in W.MODELS if m.startswith("I")]
R, P = W.walk_forward(X, [2020, 2021, 2022, 2023, 2024, 2025], ms, ens={"ens": ["standings", "logit"], "I3_ens_with_fused": ["standings", "logit", "I3_fused_strength_only"]}, tie=None)
print("\n=== PODIUM (pre-weekend, 131 races)"); report(R, "ens")
