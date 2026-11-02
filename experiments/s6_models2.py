import sys, warnings; sys.path.insert(0, ".")
from experiments.common import *
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Ck
from f1.profile import load_aux
from f1.kalman import run_kf, team_obs
warnings.filterwarnings("ignore")
X = table(); BASE = pre_weekend_cols(X); res = load_results(); P0 = pd.read_pickle("data/cache_P.pkl")
SM = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
# ================= E74 GP over (time x circuit traits) for team race pace
tel, circ, strat, ses, pits, pace = load_aux()
cm = circ.merge(res.drop_duplicates(["season", "round"])[["season", "round", "circuit"]], on=["season", "round"]); cm["fs"] = cm.c_slow / (cm.c_slow + cm.c_med + cm.c_fast)
st = strat.merge(res.drop_duplicates(["season", "round"])[["season", "round", "circuit"]], on=["season", "round"])
T = cm.groupby("circuit")[["fs", "c_ft", "c_vmax", "c_lowspeed", "c_longest"]].mean().join(st.groupby("circuit")[["stops_2plus"]].mean()); T = ((T - T.mean()) / T.std()).fillna(0)
circ_of = res.drop_duplicates(["season", "round"]).set_index(["season", "round"]).circuit
obs = team_obs(X, "y_pace"); obs["circuit"] = [circ_of[(s, r)] for s, r in zip(obs.season, obs["round"])]
kmap = X.drop_duplicates("k").set_index("k")[["season", "round"]]
def gp_forecast(use_traits, kernel_len=6.0, window=45):
    out = []
    for k in sorted(obs[obs.season.between(2022, 2025)].k.unique()):
        cur = obs[obs.k == k]; cc = cur.circuit.iloc[0]; tv = T.loc[cc].values if cc in T.index else np.zeros(T.shape[1])
        for _, r in cur.iterrows():
            p = obs[(obs.team == r.team) & (obs.k < k) & (obs.k >= k - window)]
            if len(p) < 6: continue
            Z = np.c_[p.k.values, np.vstack([T.loc[c].values if c in T.index else np.zeros(T.shape[1]) for c in p.circuit])] if use_traits else p.k.values[:, None]
            z0 = np.r_[k, tv][None, :] if use_traits else np.array([[k]])
            ker = Ck(1.0) * RBF(length_scale=[kernel_len] + [4.0] * (Z.shape[1] - 1), length_scale_bounds=(0.5, 200)) + WhiteKernel(0.3)
            y = p.y_pace.values; mu, sd = y.mean(), y.std() + 1e-6
            try: g = GaussianProcessRegressor(ker, normalize_y=False, n_restarts_optimizer=0).fit(Z, (y - mu) / sd); out.append((k, r.team, g.predict(z0)[0] * sd + mu))
            except Exception: pass
    return pd.DataFrame(out, columns=["k", "team", "f"])
def team_rho(f):
    o = obs.merge(f, on=["k", "team"]); return o.groupby("k").apply(lambda g: spearmanr(g.f, g.y_pace)[0] if len(g) > 4 else np.nan, include_groups=False).mean(), len(o)
def recency(hl=4.0):
    rows = []
    for k in sorted(obs[obs.season.between(2022, 2025)].k.unique()):
        p = obs[(obs.k < k)]; p = p[p.season >= obs[obs.k == k].season.iloc[0] - 1]; w = 0.5 ** ((k - p.k) / hl); g = (p.y_pace * w).groupby(p.team).sum() / w.groupby(p.team).sum(); rows += [(k, t, v) for t, v in g.items()]
    return pd.DataFrame(rows, columns=["k", "team", "f"])
print("=== E74 Gaussian process, team race-pace forecast, 2022-25 races (team-level Spearman vs actual; higher better)")
base = team_rho(recency()); print(f"  recency hl=4 baseline                      : {base[0]:.4f} (n={base[1]})")
g0 = team_rho(gp_forecast(False)); print(f"  GP, time kernel only (development drift)   : {g0[0]:.4f} (n={g0[1]})")
g1 = team_rho(gp_forecast(True)); print(f"  GP, time x circuit-trait kernel (dev+track): {g1[0]:.4f} (n={g1[1]})")
# ================= E76 online regime-adaptive ensemble (Hedge) over stored walk-forward scores
names = ["standings", "logit|C0.03|hl100", "ridge_pace", "et_order", "lgb_rank", "et_pace"]
Pw = P0[P0.model.isin(names)].copy(); Pw["r"] = Pw.groupby(["model", "season", "round"]).score.rank(pct=True)
wide = Pw.pivot_table(index=["season", "round", "driver"], columns="model", values="r").reset_index().merge(X[["season", "round", "driver", "order", "gridn"]], on=["season", "round", "driver"])
races = [g for _, g in wide.groupby(["season", "round"], sort=True)]
def hedge(eta):
    L = np.zeros(len(names)); out = []
    for g in races:
        w = np.exp(-eta * (L - L.min())); w /= w.sum(); sc = (g[names].values * w).sum(1); m = W.metrics(g, sc, tie=None); out.append((g.season.iloc[0], m["podium"], m["ndcg3"]))
        L += np.array([1 - W.metrics(g, g[n].values, tie=None)["ndcg3"] for n in names])   # update with this race's realised loss (used only for LATER races)
    return pd.DataFrame(out, columns=["s", "pod", "ndcg"])
print("\n=== E76 online Hedge ensemble (weights from past races' realised loss only)")
for eta in (0.5, 2, 5, 10):
    d = hedge(eta); print(f"  eta={eta:4.1f}: podium inner {d[d.s<=2022].pod.mean():.3f} outer {d[d.s>=2023].pod.mean():.3f} | ndcg3 inner {d[d.s<=2022].ndcg.mean():.3f} outer {d[d.s>=2023].ndcg.mean():.3f}")
eq = pd.DataFrame([(g.season.iloc[0], W.metrics(g, g[names].mean(1).values, tie=None)["podium"]) for g in races], columns=["s", "pod"]); print(f"  equal-weight reference: inner {eq[eq.s<=2022].pod.mean():.3f} outer {eq[eq.s>=2023].pod.mean():.3f}")
# ================= E77 dominance-conditioned: when one team is far ahead (pre-weekend t10 pace gap) trust learned logit, else standings
tp = X.groupby(["k", "team"]).t10_pace_gap.first().reset_index(); dom = tp.groupby("k").t10_pace_gap.apply(lambda s: np.sort(s.dropna().values)[1] - np.sort(s.dropna().values)[0] if s.notna().sum() > 2 else np.nan).rename("dom")
wide["k"] = wide[["season", "round"]].merge(X.drop_duplicates(["season", "round"])[["season", "round", "k"]], on=["season", "round"], how="left").k.values; wide = wide.merge(dom, on="k", how="left")
inn = wide[wide.season <= 2022].groupby(["season", "round"]).dom.first(); print("\n=== E77 dominance-conditioned model choice (threshold on dominance chosen on INNER)")
best = None
for th in np.nanquantile(inn, [.2, .35, .5, .65, .8]):
    rr = [W.metrics(g, (g["logit|C0.03|hl100"] if g.dom.iloc[0] >= th else g.standings).values, tie=None)["podium"] for _, g in wide[wide.season <= 2022].groupby(["season", "round"])]
    if best is None or np.mean(rr) > best[0]: best = (np.mean(rr), th)
th = best[1]; out = [W.metrics(g, (g["logit|C0.03|hl100"] if g.dom.iloc[0] >= th else g.standings).values, tie=None)["podium"] for _, g in wide[wide.season >= 2023].groupby(["season", "round"])]
print(f"  inner-best threshold {th:.2f} (inner podium {best[0]:.3f}) -> OUTER podium {np.mean(out):.3f}   (standings outer .590, logit outer .557)")
# ================= E80 two-layer Kalman: team level + driver offset, ranked by forecast pace
Xd = X.copy(); tm = Xd.groupby(["k", "team"]).y_pace.transform("mean"); Xd["off"] = Xd.y_pace - tm
kt = run_kf(X, "y_pace", q=0.01, qs=0.2, r=1.0).rename(columns={"f": "ft"})
def driver_kf(q=0.02, r=0.5, qs=0.3):
    st = {}; rows = []; prev = None
    for k, g in Xd.sort_values("k").groupby("k", sort=True):
        s = g.season.iloc[0]
        for c in g.code:
            m, Pv = st.get(c, [0.0, 2.0]); Pv += q + (qs if prev is not None and s != prev else 0); st[c] = [m, Pv]; rows.append((k, c, m))
        for c, o in zip(g.code, g.off):
            if np.isfinite(o): m, Pv = st[c]; K = Pv / (Pv + r); st[c] = [m + K * (o - m), (1 - K) * Pv]
        prev = s
    return pd.DataFrame(rows, columns=["k", "code", "doff"])
Xk = X.merge(kt[["k", "team", "ft"]], on=["k", "team"], how="left").merge(driver_kf(), on=["k", "code"], how="left"); Xk["fpace"] = Xk.ft + Xk.doff
W.MODELS["E80_two_layer_kalman"] = lambda tr, te, F, **kw: te.fpace.fillna(te.fpace.max()).values; W.MODELS["standings"] = SM
W.MODELS["logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, BASE, C=0.03, hl=100)
R, _ = W.walk_forward(Xk, [2020, 2021, 2022, 2023, 2024, 2025], ["standings", "logit", "E80_two_layer_kalman"], ens={"ens": ["standings", "logit"]}, tie=None)
print("\n=== E80 two-layer (team + driver-offset) Kalman forecast pace, ranked directly"); report(R, "ens")
