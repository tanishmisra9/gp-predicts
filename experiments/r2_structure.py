import sys; sys.path.insert(0, ".")
from experiments.common import *
from scipy.optimize import minimize
X = table(); BASE = pre_weekend_cols(X); DEV = [2020, 2021, 2022, 2023, 2024, 2025]
SM = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
LG = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, BASE, C=0.03, hl=100)
W.MODELS.update(standings=SM, logit=LG)
imp = lambda tr, te, F: (lambda i, s: (s.transform(i.transform(tr[F])), s.transform(i.transform(te[F]))))(*(lambda i: (i, StandardScaler().fit(i.transform(tr[F]))))(SimpleImputer(strategy="median").fit(tr[F])))
rk = lambda s: pd.Series(s).rank(pct=True).values
# ---- E28 team-then-driver: predict TEAM finishing order (best car), within-team order from the logit
def team_driver(tr, te, F, lam=1.0, **kw):
    tt = tr.groupby(["k", "team"]).agg(o=("order", "min"), **{c: (c, "mean") for c in BASE}).reset_index()
    Z = imp(tt, te.groupby("team")[BASE].mean().reset_index(), BASE) if False else None
    tte = te.groupby("team")[BASE].mean().reset_index()
    f = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=300)).fit(tt[BASE], tt.o, ridge__sample_weight=W._w(tt.assign(k=tt.k), 100))
    tr_pred = pd.Series(f.predict(tte[BASE]), index=tte.team)
    within = pd.Series(LG(tr, te, F), index=te.index).groupby(te.team).rank()
    return (te.team.map(tr_pred).rank(method="dense") * 10 + within).values
W.MODELS["E28_team_then_driver"] = team_driver
# ---- E29 diversity penalty on the weaker teammate (fix 2+1 over-concentration)
def diversity(delta):
    def f(tr, te, F, **kw):
        s = pd.Series(rk(SM(tr, te, F)) + rk(LG(tr, te, F)), index=te.index) / 2; second = s.groupby(te.team).rank(method="first") == 2
        return (s + delta * second).values
    return f
for d in (0.05, 0.1, 0.2): W.MODELS[f"E29_diversity_{d}"] = diversity(d)
# ---- E30 Plackett-Luce (top-3 sequential), linear utility, L2
def plackett(tr, te, F, l2=3.0, **kw):
    F = BASE  # explicit: never use the harness default
    A, B = imp(tr, te, F); tr = tr.reset_index(drop=True); wk = W._w(tr, 100).values
    races = [(np.where(g.index)[0] if False else g.index.values) for _, g in tr.groupby("k")]
    order = tr.order.values
    def nll(w):
        u = A @ w; L = 0.0; G = np.zeros_like(w)
        for idx in races:
            o = order[idx]; top = idx[np.argsort(o)][:3]; rem = np.ones(len(idx), bool); pos = {j: i for i, j in enumerate(idx)}
            ww = wk[idx[0]]
            for t in top:
                loc = [pos[j] for j in idx[rem]] if False else None
                break
            ids = list(idx); remaining = ids.copy()
            for t in top:
                ur = u[remaining]; m = ur.max(); e = np.exp(ur - m); Z = e.sum()
                L -= ww * (u[t] - m - np.log(Z)); G -= ww * (A[t] - (e[:, None] * A[remaining]).sum(0) / Z)
                remaining.remove(t)
        return L + l2 * (w @ w), G + 2 * l2 * w
    w = minimize(nll, np.zeros(A.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 60}).x
    return -(B @ w)
W.MODELS["E30_plackett_luce"] = plackett
# ---- E31 all-pairs Bradley-Terry with top-6 focus (pre-weekend)
def bradley(tr, te, F, C=0.02, **kw):
    F = BASE
    A, B = imp(tr, te, F); tr = tr.reset_index(drop=True); xs, ys, ws = [], [], []; wk = W._w(tr, 100).values
    for k, g in tr.groupby("k"):
        idx = g.index.values; o = g.order.values; sel = np.where(o <= 6)[0]
        for i in sel:
            for j in range(len(idx)):
                if o[i] < o[j]: xs += [A[idx[i]] - A[idx[j]]]; ys += [1]; ws += [wk[idx[i]]]
    xs = np.array(xs); X2 = np.vstack([xs, -xs]); y2 = np.r_[np.ones(len(xs)), np.zeros(len(xs))]; w2 = np.r_[ws, ws]
    m = LogisticRegression(C=C, fit_intercept=False, max_iter=300).fit(X2, y2, sample_weight=w2); return -(B @ m.coef_[0])
from sklearn.linear_model import LogisticRegression, Ridge
W.MODELS["E31_bradley_terry_pairs"] = bradley
# ---- E32 rank-transformed (within-race) features
rkcols = [c for c in BASE if c.split("_", 1)[0] in ("d3", "d10", "t3", "t10")]
for c in rkcols: X["rk_" + c] = X.groupby("k")[c].rank(pct=True)
RK = [c for c in X if c.startswith("rk_")] + ["d_season_pts", "t_season_pts", "prev_order", "season_races"]
X["rk_dpts"] = X.groupby("k").d_season_pts.rank(pct=True); RK.append("rk_dpts")
W.MODELS["E32_rank_features_logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, RK, C=0.03, hl=100)
W.MODELS["E32b_rank_features_lgb"] = lambda tr, te, F, **kw: W.m_lgb_pod(tr, te, RK, hl=100)
ms = ["standings", "logit"] + [m for m in W.MODELS if m.startswith("E")]
R, P = W.walk_forward(X, DEV, ms, ens={"ens": ["standings", "logit"]}, tie=None)
report(R, "ens")
P.to_pickle("data/cache_P_r2.pkl")
