import sys; sys.path.insert(0, ".")
from experiments.common import *
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
X = table(); BASE = pre_weekend_cols(X); P0 = pd.read_pickle("data/cache_P.pkl")
pw = P0[P0.model.isin(["standings", "logit|C0.03|hl100"])].copy(); pw["r"] = pw.groupby(["model", "season", "round"]).score.rank(pct=True)
ens = pw.groupby(["season", "round", "driver"]).r.mean().rename("ens").reset_index(); st = pw[pw.model == "standings"][["season", "round", "driver", "r"]].rename(columns={"r": "st"})
D = X[X.season.between(2020, 2025)].merge(ens, on=["season", "round", "driver"]).merge(st, on=["season", "round", "driver"])
def fam(g, col):
    s = g.sort_values([col, "driver"]); tr = set(g[g.order <= 3].driver); top = lambda n: set(s.head(n).driver)
    return dict(strict_overlap=len(top(3) & tr) / 3, at_least_2=float(len(top(3) & tr) >= 2), top4_cover=len(top(4) & tr) / 3, top5_cover=len(top(5) & tr) / 3, winner_top2=float(s.sort_values(col).head(2).driver.isin(g[g.order == 1].driver).any()),
                binary_acc=1 - len(top(3) ^ tr) / len(g))
rows = []
for (sn, rn), g in D.groupby(["season", "round"]):
    for m in ("ens", "st"): rows.append(dict(season=sn, round=rn, model=m, **fam(g, m)))
F = pd.DataFrame(rows); rng = np.random.default_rng(0)
print("=== E88 FAMILY OF TARGETS with 95% bootstrap CI (131 races, 2020-25)")
for col in ("strict_overlap", "at_least_2", "top4_cover", "top5_cover", "winner_top2", "binary_acc"):
    for m in ("ens", "st"):
        v = F[F.model == m][col].values; b = rng.choice(v, (4000, len(v))).mean(1); print(f"  {col:15s} {m:4s} {v.mean():.3f}  [{np.percentile(b,2.5):.3f}, {np.percentile(b,97.5):.3f}]")
print("\n=== E89 ANATOMY: can we predict WHICH races the model nails? (race-level AUC, leave-one-season-out)")
R = F[F.model == "ens"].set_index(["season", "round"]).strict_overlap
Dr = X[X.season.between(2020, 2025)].drop_duplicates(["season", "round"]).set_index(["season", "round"])
Dr["dom"] = X.groupby("k").t10_pace_gap.apply(lambda s: np.sort(s.dropna().values)[1] - np.sort(s.dropna().values)[0] if s.notna().sum() > 2 else np.nan).reindex(Dr.k.values).values
desc = [c for c in ["dom", "season_races", "stops_2plus", "c_chaos", "c_dnf", "c_coll", "sc_frac", "circ_grid_corr", "circ_pole_win", "field", "wet"] if c in Dr]
Y = (R >= .667).astype(int).reindex(Dr.index); Z = Dr[desc].fillna(Dr[desc].median()); preds = pd.Series(0.0, index=Y.index)
for s in Dr.index.get_level_values(0).unique():
    te = Dr.index.get_level_values(0) == s; m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3)).fit(Z[~te], Y[~te]); preds[te] = m.predict_proba(Z[te])[:, 1]
print(f"  predicting 'race podium >= 2/3' from {desc}: leave-one-season-out AUC = {roc_auc_score(Y, preds):.3f}  (0.5 = no better than chance); base rate {Y.mean():.2f}")
print("\n=== E90 LEAVE-ONE-BLOCK-OUT ablation of the logit (outer podium 2023-25 / inner), full BASE = reference")
blocks = {"d3_*": [c for c in BASE if c.startswith("d3_")], "d10_*": [c for c in BASE if c.startswith("d10_")], "t3_*": [c for c in BASE if c.startswith("t3_")], "t10_*": [c for c in BASE if c.startswith("t10_")],
          "season standings": ["d_season_pts", "t_season_pts", "season_races"], "circuit stats": [c for c in BASE if c.startswith(("circ_", "t_circ", "d_circ"))], "prev_order": ["prev_order"], "era/field": ["era_round", "field"]}
for nm, cols in {"FULL": []}.items(): pass
W.MODELS["FULL"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, BASE, C=0.03, hl=100)
for nm, cols in blocks.items(): W.MODELS["-" + nm] = (lambda cols: lambda tr, te, F, **kw: W.m_logit_pod(tr, te, [c for c in BASE if c not in cols], C=0.03, hl=100))(cols)
R2, _ = W.walk_forward(X, [2020, 2021, 2022, 2023, 2024, 2025], list(W.MODELS) if False else ["FULL"] + ["-" + n for n in blocks], tie=None); report(R2, "FULL")
print("\n=== E91 SIGN TEST ens vs standings per race (2020-25)")
a = F[F.model == "ens"].set_index(["season", "round"]).strict_overlap; b = F[F.model == "st"].set_index(["season", "round"]).strict_overlap; w, l = int((a > b).sum()), int((a < b).sum())
print(f"  ens better in {w} races, standings better in {l}, tied {int((a == b).sum())}; sign test p = {binomtest(w, w + l, .5).pvalue:.3f}")
print("  by season (ens - standings):", (a - b).groupby(level=0).mean().round(3).to_dict())
print("\n=== E92 STICKINESS: does the predicted podium change when the real podium changes?")
pred = D.sort_values(["ens", "driver"]).groupby(["season", "round"]).head(3).groupby(["season", "round"]).driver.apply(set); act = D[D.order <= 3].groupby(["season", "round"]).driver.apply(set)
jac = lambda s: [len(s.iloc[i] & s.iloc[i - 1]) / len(s.iloc[i] | s.iloc[i - 1]) for i in range(1, len(s))]
jp, ja = np.array(jac(pred)), np.array(jac(act)); print(f"  consecutive-race podium overlap (Jaccard): predicted {jp.mean():.2f} vs actual {ja.mean():.2f};  corr(change in prediction, change in reality) = {np.corrcoef(jp, ja)[0,1]:+.3f}")
