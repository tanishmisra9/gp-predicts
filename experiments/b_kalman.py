import sys; sys.path.insert(0, ".")
import itertools
from experiments.common import *
from scipy.stats import spearmanr
from f1.kalman import run_kf, team_obs
X = table(cache=False)
DEV = X[X.season.between(2020, 2025)]
def team_rho(col, f):
    """per-race Spearman between forecast and the actual team observation (lower=faster both)"""
    o = team_obs(X, col).merge(f, on=["k", "team"]); o = o[o.season.between(2020, 2025)]
    return o.groupby("k").apply(lambda g: spearmanr(g.f, g[col])[0] if len(g) > 4 else np.nan, include_groups=False).mean(), np.sqrt(((o.f - o[col]) ** 2).mean())
# baseline: recency-weighted team mean (hl=4 races, current+previous season)
def recency(col, hl=4.0):
    o = team_obs(X, col); rows = []
    for k in sorted(o.k.unique()):
        p = o[(o.k < k)]; p = p[p.season >= o[o.k == k].season.iloc[0] - 1]
        w = 0.5 ** ((k - p.k) / hl); g = (p[col] * w).groupby(p.team).sum() / w.groupby(p.team).sum()
        rows += [(k, t, v) for t, v in g.items()]
    return pd.DataFrame(rows, columns=["k", "team", "f"])
for col in ("y_pace", "q_gap"):
    print(f"\n=== {col}: team-level forecast quality, 2020-25 (Spearman higher=better | RMSE lower=better)")
    print("  recency hl=4   : %.4f | %.3f" % team_rho(col, recency(col)))
    best = None
    for q, qs, r in itertools.product((0.01, 0.03, 0.1), (0.2, 0.6, 1.5), (0.3, 1.0)):
        f = run_kf(X, col, q=q, qs=qs, r=r); rho, rmse = team_rho(col, f)
        if best is None or rho > best[0]: best = (rho, rmse, q, qs, r)
    print("  KALMAN best    : %.4f | %.3f   (q=%.2f qs=%.1f r=%.1f)" % best)
