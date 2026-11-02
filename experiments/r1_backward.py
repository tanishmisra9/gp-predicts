import sys; sys.path.insert(0, ".")
from experiments.common import *
from sklearn.metrics import roc_auc_score
X = table(); P = pd.read_pickle("data/cache_P.pkl"); res = load_results()
X = X.merge(res[["season", "round", "code", "status"]], on=["season", "round", "code"], how="left")
X["fin"] = X.status.isin(["Finished", "Lapped"]) | X.status.str.startswith("+")
pw = P[P.model.isin(["standings", "logit|C0.03|hl100"])].copy(); pw["r"] = pw.groupby(["model", "season", "round"]).score.rank(pct=True)
ens = pw.groupby(["season", "round", "driver"]).r.mean().rename("ens").reset_index()
pr = pw[pw.model == "logit|C0.03|hl100"][["season", "round", "driver", "score"]].rename(columns={"score": "negp"})
D = X[X.season.between(2020, 2025)].merge(ens, on=["season", "round", "driver"]).merge(pr, on=["season", "round", "driver"])
D["split"] = np.where(D.season <= 2022, "inner", "outer")
def ov(g, sc):  # podium overlap with ties by driver name
    o = g.assign(s=sc).sort_values(["s", "driver"]).driver.head(3); return len(set(o) & set(g[g.order <= 3].driver)) / 3
def lab(name, f):
    r = D.groupby(["season", "round"]).apply(lambda g: ov(g, f(g)), include_groups=False); s = D.drop_duplicates(["season", "round"]).set_index(["season", "round"]).split
    print(f"  {name:62s} all {r.mean():.3f} | inner {r[s=='inner'].mean():.3f} outer {r[s=='outer'].mean():.3f}")
print("=== E22 ORACLE LADDER (what does each unknown cost?)  [frozen ens = reference]")
lab("ens (reference)", lambda g: g.ens.values)
lab("ens, but KNOWING who retires (retirers excluded)", lambda g: np.where(g.fin, g.ens, 9))
def team_true(g):   # true team order (team's best finisher), within-team order from ens
    tr = g.groupby("team").order.transform("min"); return (tr.rank(method="dense") * 10 + g.groupby("team").ens.rank()).values
lab("ens within team, but KNOWING true TEAM ORDER", team_true)
def mate_true(g):   # keep ens team slots, assign them to drivers by true teammate order
    s = g.ens.copy()
    for t, h in g.groupby("team"):
        if len(h) == 2:
            lo, hi = sorted(h.ens.values); d = h.sort_values("order").index; s.loc[d[0]] = lo; s.loc[d[1]] = hi
    return s.values
lab("ens, but KNOWING true TEAMMATE ORDER", mate_true)
lab("KNOWING team order AND teammate order (true finishing order)", lambda g: g.order.values)
def pace_true(g): return g.y_pace.fillna(99).values
lab("KNOWING each driver's true race PACE (oracle)", pace_true)
def teampace(g):
    tp = g.groupby("team").y_pace.transform("mean").fillna(99); return (tp.rank(method="dense") * 10 + g.groupby("team").ens.rank()).values
lab("ens within team, KNOWING true TEAM race pace", teampace)
print("\n=== E23 HINDSIGHT fixed-effect oracles (uses the whole season INCLUDING FUTURE races, minus the race itself)")
n = D.groupby(["season", "driver"]).order.transform("count"); D["loo_drv"] = (D.groupby(["season", "driver"]).order.transform("sum") - D.order) / (n - 1)
tm = D.groupby(["season", "team"]).order.transform("sum"); nt = D.groupby(["season", "team"]).order.transform("count"); D["loo_team"] = (tm - D.order) / (nt - 1)
lab("rank by driver's season-mean finish (LOO, hindsight)", lambda g: g.loo_drv.fillna(99).values)
lab("rank by TEAM season-mean finish (LOO) then ens within team", lambda g: (g.loo_team.rank(method="dense") * 10 + g.groupby("team").ens.rank()).values)
print("\n=== E24 within-race AUC of single pre-weekend features for 'on podium' (inner-picked, outer-confirmed)")
feats = [c for c in pre_weekend_cols(X) if c in D and D[c].notna().mean() > .6]
rows = []
for c in feats:
    for sp in ("inner", "outer"):
        d = D[D.split == sp]; a = []
        for _, g in d.groupby(["season", "round"]):
            v = g[c]; 
            if v.notna().sum() > 5 and g.podium.nunique() == 2: a.append(roc_auc_score(g.podium, -v.fillna(v.median())))
        rows.append((c, sp, np.mean(a)))
A = pd.DataFrame(rows, columns=["f", "sp", "auc"]).pivot(index="f", columns="sp", values="auc"); A["dir_inner"] = np.sign(A.inner - .5)
A["outer_signed"] = np.where(A.dir_inner > 0, A.outer, 1 - A.outer); A["inner_signed"] = np.where(A.dir_inner > 0, A.inner, 1 - A.inner)
print(A.sort_values("inner_signed", ascending=False).head(10)[["inner_signed", "outer_signed"]].round(3).to_string())
print("\n=== E25 PODIUM PATTERN (how many podium slots does one team take?)")
pat = D[D.podium == 1].groupby(["season", "round", "team"]).size().groupby(["season", "round"]).apply(lambda s: "2+1" if (s == 2).any() else "1+1+1").value_counts(normalize=True)
print("  actual:", pat.round(3).to_dict())
def pred_pat(g):
    t = g.sort_values("ens").head(3).team.value_counts(); return "2+1" if (t == 2).any() else "1+1+1"
pp = D.groupby(["season", "round"]).apply(pred_pat, include_groups=False); print("  predicted:", pp.value_counts(normalize=True).round(3).to_dict())
ac = D[D.podium == 1].groupby(["season", "round", "team"]).size().groupby(["season", "round"]).apply(lambda s: "2+1" if (s == 2).any() else "1+1+1")
print("  predicted pattern == actual pattern: %.3f" % (pp == ac).mean())
print("\n=== E26 SELECTIVE PREDICTION: accuracy when the model is confident (confidence = sum of its top-3 podium probabilities)")
rows = []
for (s, r), g in D.groupby(["season", "round"]):
    top = g.sort_values("negp").head(4); p = -top.negp.values; rows.append((s, r, p[:3].sum(), p[2] - p[3], ov(g, g.ens.values)))
C = pd.DataFrame(rows, columns=["s", "r", "conf_sum", "conf_gap", "acc"]); C["split"] = np.where(C.s <= 2022, "inner", "outer")
for col in ("conf_sum", "conf_gap"):
    q = pd.qcut(C[col], 4, labels=["lowest 25%", "25-50%", "50-75%", "most confident 25%"]); print(f"  by {col}:", C.groupby(q, observed=True).acc.mean().round(3).to_dict(), " corr(conf,acc)=%.3f" % C[col].corr(C.acc))
thr = C[C.split == "inner"].conf_sum.quantile(.75); o = C[C.split == "outer"]
print(f"  threshold chosen on INNER (top 25% most confident): OUTER accuracy on confident races = {o[o.conf_sum >= thr].acc.mean():.3f} (coverage {(o.conf_sum >= thr).mean():.0%}, n={int((o.conf_sum >= thr).sum())}); rest {o[o.conf_sum < thr].acc.mean():.3f}")
print("\n=== E27 Is race difficulty predictable from pre-race descriptors?  corr(podium overlap, descriptor):")
Dd = D.drop_duplicates(["season", "round"]).merge(C[["s", "r", "acc"]].rename(columns={"s": "season", "r": "round"}), on=["season", "round"])
for c in ["season_races", "c_chaos", "c_dnf", "c_coll", "sc_frac", "stops_2plus", "field", "c_pole_win" if "c_pole_win" in Dd else "circ_pole_win", "circ_grid_corr"]:
    if c in Dd: print(f"  {c:16s} {Dd[c].corr(Dd.acc):+.3f}")
