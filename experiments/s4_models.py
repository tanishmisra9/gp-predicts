import sys, warnings; sys.path.insert(0, ".")
from experiments.common import *
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Ck
from f1.profile import load_aux
X = table(); BASE = pre_weekend_cols(X); DEV = [2020, 2021, 2022, 2023, 2024, 2025]; res = load_results()
res["dnf_y"] = (~(res.status.isin(["Finished", "Lapped"]) | res.status.str.startswith("+")) & ~res.status.isin(["Did not start", "Withdrew"])).astype(float)
X = X.merge(res[["season", "round", "code", "dnf_y", "status", "points"]], on=["season", "round", "code"], how="left")
W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
W.MODELS["logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, BASE, C=0.03, hl=100)
# ---- E75 Monte Carlo with a TEAM-SHARED shock (strategy / reliability / safety-car luck is shared by both cars)
def mc_team(sigma_d=0.5, sigma_t=0.7, S=4000):
    def f(tr, te, F, **kw):
        mu = reg(tr, te, BASE, "y_pace", "ridge", alpha=500); rng = np.random.default_rng(int(te.k.iloc[0]))
        DNF = [c for c in ["d10_dnf", "t10_dnf", "d3_dnf", "t3_dnf", "d_coll", "c_dnf", "c_coll", "pu_pufail", "era_round"] if c in X]
        t = tr[tr.dnf_y.notna()]; lg = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=400)).fit(t[DNF], t.dnf_y, logisticregression__sample_weight=W._w(t, 100))
        pdn = lg.predict_proba(te[DNF])[:, 1]; sp = np.std(mu); teams = pd.factorize(te.team)[0]; nt = teams.max() + 1
        zt = rng.normal(size=(S, nt))[:, teams]; zd = rng.normal(size=(S, len(te)))
        perf = mu[None, :] + sp * (sigma_d * zd + sigma_t * zt); perf = np.where(rng.random((S, len(te))) < pdn[None, :], np.inf, perf)
        top3 = np.argpartition(perf, 2, axis=1)[:, :3]; p = np.zeros(len(te))
        for j in range(3): np.add.at(p, top3[:, j], 1)
        return -(p / S)
    return f
for sd, st in ((0.5, 0.0), (0.5, 0.4), (0.4, 0.8)): W.MODELS[f"E75_mc_team_shock_d{sd}_t{st}"] = mc_team(sd, st)
# ---- E78 multinomial position bins -> P(podium)
bins = lambda o: np.select([o == 1, o <= 3, o <= 6, o <= 10], [0, 1, 2, 3], 4)
def multinom(tr, te, F, **kw):
    y = bins(tr.order.values); p = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.03, max_iter=400)).fit(tr[BASE], y, logisticregression__sample_weight=W._w(tr, 100))
    pr = p.predict_proba(te[BASE]); cl = list(p.classes_); win = pr[:, cl.index(0)] if 0 in cl else 0; p23 = pr[:, cl.index(1)] if 1 in cl else 0
    return -(win + p23)
W.MODELS["E78_multinomial_bins"] = multinom
# ---- E79 circuit-cluster conditioned form: team form restricted to past races at SIMILAR circuits (k-means on layout traits)
tel, circ, strat, ses, pits, pace = load_aux()
cm = circ.merge(res.drop_duplicates(["season", "round"])[["season", "round", "circuit"]], on=["season", "round"]); cm["fs"] = cm.c_slow / (cm.c_slow + cm.c_med + cm.c_fast)
T = cm.groupby("circuit")[["fs", "c_ft", "c_vmax", "c_lowspeed", "c_longest"]].mean(); T = (T - T.mean()) / T.std(); cl = pd.Series(KMeans(4, n_init=10, random_state=0).fit_predict(T.fillna(0)), index=T.index)
cc = res.drop_duplicates(["season", "round"])[["season", "round", "circuit"]]; cc["cl"] = cc.circuit.map(cl); X = X.merge(cc[["season", "round", "cl"]], on=["season", "round"], how="left")
tp = X.groupby(["k", "team"]).agg(y=("y_pace", "mean"), cl=("cl", "first"), season=("season", "first")).reset_index()
out = []
for k in sorted(X.k.unique()):
    c0 = X[X.k == k].cl.iloc[0]
    if pd.isna(c0): continue
    p_ = tp[(tp.k < k) & (tp.cl == c0)].dropna(subset=["y"]); 
    if p_.empty: continue
    w = 0.5 ** ((k - p_.k) / 20); g = (p_.y * w).groupby(p_.team).sum() / w.groupby(p_.team).sum(); out.append(pd.DataFrame({"k": k, "team": g.index, "t_clpace": g.values}))
X = X.merge(pd.concat(out), on=["k", "team"], how="left")
W.MODELS["E79_cluster_conditioned_form"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, BASE + ["t_clpace"], C=0.03, hl=100)
# ---- E81 race-trim strength: decayed (race pace gap - quali gap) per team
X["rq"] = X.y_pace - X.q_gap
def dec(col, key, hl):
    out = []
    for k in sorted(X.k.unique()):
        p_ = X[(X.k < k)].dropna(subset=[col]); 
        if p_.empty: continue
        w = 0.5 ** ((k - p_.k) / hl); g = (p_[col] * w).groupby(p_[key]).sum() / w.groupby(p_[key]).sum(); out.append(pd.DataFrame({"k": k, key: g.index, "v": g.values}))
    return pd.concat(out)
X = X.merge(dec("rq", "team", 8).rename(columns={"v": "t_racetrim"}), on=["k", "team"], how="left").merge(dec("rq", "code", 8).rename(columns={"v": "d_racetrim"}), on=["k", "code"], how="left")
W.MODELS["E81_race_trim_strength"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, BASE + ["t_racetrim", "d_racetrim"], C=0.03, hl=100)
# ---- E82 sprint augmentation: add sprint races as extra labelled races (label = sprint podium), half weight
def sprint_aug(tr, te, F, **kw):
    sp = tr[tr.s_order.notna()].copy(); sp["podium"] = (sp.s_order <= 3).astype(int); T = pd.concat([tr, sp]); w = np.r_[W._w(tr, 100), 0.5 * W._w(sp, 100)]
    p = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.03, max_iter=500)).fit(T[BASE], T.podium, logisticregression__sample_weight=w); return -p.predict_proba(te[BASE])[:, 1]
W.MODELS["E82_sprint_augmentation"] = sprint_aug
ms = ["standings", "logit"] + [m for m in W.MODELS if m.startswith("E7") or m.startswith("E8")]
R, P = W.walk_forward(X, DEV, ms, ens={"ens": ["standings", "logit"]}, tie=None); report(R, "ens"); P.to_pickle("data/cache_P_s4.pkl")
