import sys; sys.path.insert(0, ".")
from experiments.common import *
from scipy.stats import spearmanr
X = table(); P = pd.read_pickle("data/cache_P.pkl")
pw = P[P.model.isin(["standings", "logit|C0.03|hl100"])].copy(); pw["r"] = pw.groupby(["model", "season", "round"]).score.rank(pct=True)
ens = pw.groupby(["season", "round", "driver"]).r.mean().rename("ens").reset_index()
pr = pw[pw.model == "logit|C0.03|hl100"][["season", "round", "driver", "score"]].rename(columns={"score": "negp"})
D = X[X.season.between(2020, 2025)].merge(ens, on=["season", "round", "driver"]).merge(pr, on=["season", "round", "driver"]); print("rows with NaN prob:", int(D.negp.isna().sum()))
rng = np.random.default_rng(0)
print("=== E56 HOW GOOD MUST TEAM-ORDER PREDICTION BE?  (true team order corrupted with noise; within-team order from our ens)")
races = [g for _, g in D.groupby(["season", "round"])]
def podium_with_team_rank(g, tr):
    s = tr.rank(method="dense") * 10 + g.groupby("team").ens.rank(); o = g.assign(s=s.values).sort_values(["s", "driver"]).driver.head(3); return len(set(o) & set(g[g.order <= 3].driver)) / 3
for sd in (0, .25, .5, .75, 1.0, 1.5, 2.5):
    pod, rho = [], []
    for g in races:
        t = g.groupby("team").order.min(); noisy = t.rank() + rng.normal(0, sd, len(t)) * t.rank().std(); pod.append(podium_with_team_rank(g, g.team.map(noisy))); rho.append(spearmanr(t.rank(), noisy)[0])
    print(f"  noise sd={sd:4.2f}  team-order Spearman={np.mean(rho):.3f}  ->  podium accuracy {np.mean(pod):.3f}")
mine = np.mean([spearmanr(g.groupby("team").order.min().rank(), g.groupby("team").ens.mean().rank())[0] for g in races])
print(f"  MY predicted team order vs actual (best-car) team order: Spearman = {mine:.3f}")
print("\n=== E57 SELECTIVE PREDICTION (fixed): accuracy when confident; confidence = sum of top-3 predicted podium probabilities")
rows = []
for g in races:
    top = g.dropna(subset=["negp"]).sort_values("negp").head(4); p = -top.negp.values
    o = g.sort_values(["ens", "driver"]).driver.head(3); rows.append((g.season.iloc[0], p[:3].sum(), p[2] - p[3], len(set(o) & set(g[g.order <= 3].driver)) / 3))
C = pd.DataFrame(rows, columns=["season", "conf", "gap", "acc"]); C["inner"] = C.season <= 2022
for cov in (1.0, .6, .4, .25, .15):
    th = C.conf.quantile(1 - cov); s = C[C.conf >= th]; print(f"  coverage {cov:4.0%}: accuracy {s.acc.mean():.3f} (n={len(s)})")
th = C[C.inner].conf.quantile(.6); o = C[~C.inner]
print(f"  NESTED: threshold = inner top-40%; OUTER confident races: acc {o[o.conf >= th].acc.mean():.3f} (n={(o.conf >= th).sum()}), the rest {o[o.conf < th].acc.mean():.3f};  corr(conf, acc)={C.conf.corr(C.acc):+.3f}")
