import sys; sys.path.insert(0, ".")
from experiments.common import *
from sklearn.linear_model import LogisticRegression
X = table(); BASE = pre_weekend_cols(X); res = load_results()
res["dnf_y"] = (~(res.status.isin(["Finished", "Lapped"]) | res.status.str.startswith("+")) & ~res.status.isin(["Did not start", "Withdrew"])).astype(float)
X = X.merge(res[["season", "round", "code", "dnf_y"]], on=["season", "round", "code"], how="left")
DNF_F = [c for c in ["d10_dnf", "t10_dnf", "d3_dnf", "t3_dnf", "d_coll", "c_dnf", "c_coll", "pu_pufail", "era_round"] if c in X]
DEV = [2020, 2021, 2022, 2023, 2024, 2025]; S = 4000; rng = np.random.default_rng(0)
store = {}
def mc(tr, te, F, sigma=0.5, **kw):
    mu = reg(tr, te, BASE, "y_pace", "ridge", alpha=500)          # predicted race-pace gap % (lower = faster)
    t = tr[tr.dnf_y.notna()]
    lg = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=500)).fit(t[DNF_F], t.dnf_y, logisticregression__sample_weight=W._w(t, 100))
    pd_ = lg.predict_proba(te[DNF_F])[:, 1]
    z = rng.normal(size=(S, len(te))); perf = mu[None, :] + sigma * z * np.std(mu)       # noise scaled to the spread of predicted pace
    perf = np.where(rng.random((S, len(te))) < pd_[None, :], np.inf, perf)
    top3 = np.argpartition(perf, 2, axis=1)[:, :3]; p = np.zeros(len(te))
    for j in range(3): np.add.at(p, top3[:, j], 1)
    p /= S; store[(int(te.season.iloc[0]), int(te["round"].iloc[0]), sigma)] = (te.driver.values, p, te.podium.values, pd_)
    return -p
for sg in (0.3, 0.6, 1.0, 1.5): W.MODELS[f"mc_s{sg}"] = lambda tr, te, F, sg=sg, **kw: mc(tr, te, F, sigma=sg)
W.MODELS["lg_podium"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, BASE, C=0.03, hl=100)
W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
R, P = W.walk_forward(X, DEV, ["standings", "lg_podium"] + [f"mc_s{s}" for s in (0.3, 0.6, 1.0, 1.5)], tie=None)
s = W.summarize(R)[["winner", "podium", "ndcg3", "pairwise", "mae"]]
# Brier of P(podium) over all driver-races, split inner (2020-22) / outer (2023-25)
for sg in (0.3, 0.6, 1.0, 1.5):
    b = {"in": [], "out": []}
    for (sn, rd, sgm), (dr, p, y, pdn) in store.items():
        if sgm == sg: b["in" if sn <= 2022 else "out"] += list((p - y) ** 2)
    s.loc[f"mc_s{sg}", "brier_2020-22"] = np.mean(b["in"]); s.loc[f"mc_s{sg}", "brier_2023-25"] = np.mean(b["out"])
print(s.to_string())
print("avg modelled retirement probability:", round(float(np.mean([pdn.mean() for *_, (_, _, _, pdn) in store.items()])), 3), " actual DNF rate:", round(X.dnf_y.mean(), 3))
