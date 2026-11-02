import sys; sys.path.insert(0, ".")
from experiments.common import *
X = table(); BASE = pre_weekend_cols(X); P = pd.read_pickle("data/cache_P.pkl")
SM = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
print("=== S1 REGIME SHIFT vs OVERFITTING: per-season podium of logit vs standings (all trained on everything before)")
w = P[P.model.isin(["standings", "logit|C0.03|hl100"])].merge(X[["season", "round", "driver", "order"]], on=["season", "round", "driver"])
rows = []
for (m, s, r), g in w.groupby(["model", "season", "round"]):
    o = g.sort_values(["score", "driver"]).driver.head(3); rows.append((m, s, len(set(o) & set(g[g.order <= 3].driver)) / 3))
t = pd.DataFrame(rows, columns=["m", "s", "p"]).groupby(["s", "m"]).p.mean().unstack().round(3); t["logit-standings"] = (t["logit|C0.03|hl100"] - t["standings"]).round(3)
ntr = X.groupby("season").k.nunique().cumsum().shift(1); t["train_races"] = ntr.reindex(t.index).values; print(t.to_string())
print("  corr(training size, logit-standings) = %.2f  (negative = more data hurts = NOT classic overfitting)" % t["train_races"].corr(t["logit-standings"]))
print("\n=== S1b equal-length rolling training windows (logit trained on the LAST N races only)")
def roll(N):
    def f(tr, te, F, **kw):
        t = tr[tr.k >= te.k.iloc[0] - N]; return W.m_logit_pod(t, te, BASE, C=0.03, hl=100)
    return f
for N in (44, 66, 110): W.MODELS[f"logit_last{N}"] = roll(N)
W.MODELS["standings"] = SM
R, _ = W.walk_forward(X, [2020, 2021, 2022, 2023, 2024, 2025], ["standings", "logit_last44", "logit_last66", "logit_last110"], tie=None)
R["sp"] = np.where(R.season <= 2022, "inner 20-22", "outer 23-25"); print(R.groupby(["model", "sp"]).podium.mean().unstack().round(3).to_string())
print("\n=== S2 POWER: smallest true podium-accuracy gain detectable (80% power, paired, alpha .05)")
sd = R[R.model == "standings"].podium.std(); 
for n in (70, 131, 150, 250): print(f"  n={n:4d} races: per-race sd {sd:.3f} (unpaired) -> MDE ~ {2.8 * sd / np.sqrt(n):.3f} (paired diff sd is smaller; treat as upper bound)")
a = R[R.model == "logit_last66"].set_index(["season", "round"]).podium; b = R[R.model == "standings"].set_index(["season", "round"]).podium; dsd = (a - b).std(); print(f"  paired-diff sd measured {dsd:.3f} -> MDE n=70: {2.8*dsd/np.sqrt(70):.3f}, n=131: {2.8*dsd/np.sqrt(131):.3f}")
