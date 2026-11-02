import sys, glob; sys.path.insert(0, ".")
from experiments.common import *
from scipy.stats import spearmanr
X = table(); res = load_results()
tst = pd.concat([pd.read_csv(f) for f in glob.glob("data/parts/testing_*.csv")]); print("testing rows:", len(tst), "seasons", sorted(tst.season.unique()))
r1 = res[res["round"] == 1][["season", "code", "team"]]; t = tst.merge(r1, on=["season", "code"])
tt = t.groupby(["season", "team"])[["best_gap", "lr_gap"]].mean().reset_index()
act = X[X["round"] <= 3].groupby(["season", "team"]).y_pace.mean().reset_index().rename(columns={"y_pace": "act"})            # actual team race pace, rounds 1-3
prev = X.groupby(["season", "team"]).y_pace.mean().reset_index(); prev["season"] += 1; prev = prev.rename(columns={"y_pace": "prev_year"})
D = tt.merge(act, on=["season", "team"]).merge(prev, on=["season", "team"], how="left")
print("\n=== E86 team-pace-order Spearman vs ACTUAL rounds 1-3 race pace (per season, then mean)")
rows = []
for s, g in D.groupby("season"):
    f = lambda c: spearmanr(g[c], g.act, nan_policy="omit")[0]
    comb = g[["best_gap", "lr_gap"]].rank().mean(axis=1) + g.prev_year.rank(); rows.append((s, len(g), f("best_gap"), f("lr_gap"), f("prev_year"), spearmanr(comb, g.act, nan_policy="omit")[0]))
R = pd.DataFrame(rows, columns=["season", "teams", "testing_best", "testing_longrun", "last_year_pace", "testing+last_year"]).set_index("season"); print(R.round(3).to_string()); print("  mean:", R.mean().round(3).to_dict())
print("\n=== E87 early-season (rounds 1-3) podium: frozen ens vs ens blended with testing-based team strength")
P0 = pd.read_pickle("data/cache_P.pkl"); pw = P0[P0.model.isin(["standings", "logit|C0.03|hl100"])].copy(); pw["r"] = pw.groupby(["model", "season", "round"]).score.rank(pct=True)
ens = pw.groupby(["season", "round", "driver"]).r.mean().rename("ens").reset_index()
E = X[X["round"] <= 3].merge(ens, on=["season", "round", "driver"]).merge(t.groupby(["season", "code"])[["best_gap", "lr_gap"]].mean().reset_index().merge(r1, on=["season", "code"]).groupby(["season", "team"])[["best_gap", "lr_gap"]].mean().add_prefix("tt_").reset_index(), on=["season", "team"], how="left")
for w in (0, .25, .5, 1.0):
    out = []
    for (s, r), g in E.groupby(["season", "round"]):
        tr = (g.tt_lr_gap.rank(pct=True).fillna(.5) + g.tt_best_gap.rank(pct=True).fillna(.5)) / 2; sc = (1 - w) * g.ens.values + w * tr.values
        out.append((s, W.metrics(g, sc, tie=None)["podium"]))
    d = pd.DataFrame(out, columns=["s", "p"]); print(f"  testing weight {w:4.2f}: podium on rounds 1-3 = {d.p.mean():.3f} (n={len(d)} races)  by season {d.groupby('s').p.mean().round(2).to_dict()}")
