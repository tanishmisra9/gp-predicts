"""E156: frozen rounds 1-3 testing-pace blend. Pre-weekend only, no 2026 rows."""
import glob
import numpy as np
import pandas as pd
import f1.walkforward as W
from experiments.common import table
from f1.data import load_results
from f1.features import pre_weekend_cols


def main():
    X = table(); X = X[X.season <= 2025]; res = load_results()
    base = pre_weekend_cols(X)
    tst = pd.concat([pd.read_csv(f) for f in glob.glob("data/parts/testing_*.csv")])
    print("testing seasons:", sorted(tst.season.unique()))
    r1 = res[res["round"] == 1][["season", "code", "team"]]
    tt = (tst.groupby(["season", "code"])[["best_gap", "lr_gap"]].mean().reset_index().merge(r1, on=["season", "code"])
          .groupby(["season", "team"])[["best_gap", "lr_gap"]].mean().add_prefix("tt_").reset_index())
    W.MODELS["standings"] = lambda tr, te, f, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
    W.MODELS["logit"] = lambda tr, te, f, **kw: W.m_logit_pod(tr, te, base, C=0.03, hl=100)
    R, P = W.walk_forward(X, range(2020, 2026), ["standings", "logit"], ens={"ens": ["standings", "logit"]}, tie=None)
    Xt = X.merge(tt, on=["season", "team"], how="left")
    ens = P[P.model == "ens"].rename(columns={"score": "ens"})[["season", "round", "driver", "ens"]]
    rows = []
    for (s, r), g in Xt.merge(ens, on=["season", "round", "driver"]).groupby(["season", "round"]):
        have = g.tt_best_gap.notna().any() and r <= 3
        sc = g.ens.values
        if have:
            ts = (g.tt_lr_gap.rank(pct=True).fillna(.5) + g.tt_best_gap.rank(pct=True).fillna(.5)) / 2
            sc = 0.75 * g.ens.values + 0.25 * ts.values
        rows.append(dict(season=s, round=r, covered=have, ens=W.metrics(g, g.ens.values, None)["podium"],
                         blend=W.metrics(g, sc, None)["podium"]))
    D = pd.DataFrame(rows).merge(R[R.model == "standings"][["season", "round", "podium"]].rename(columns={"podium": "standings"}))
    c = D[D.covered]
    print("covered races by season:", c.groupby("season").size().to_dict())
    print("rounds 1-3 covered: ens %.3f blend %.3f standings %.3f (n=%d)" % (c.ens.mean(), c.blend.mean(), c.standings.mean(), len(c)))
    print("by season (ens, blend, standings):\n", c.groupby("season")[["ens", "blend", "standings"]].mean().round(3).to_string())
    print("blend-ens, covered:", W.paired_boot(c.blend, c.ens)[:2])
    print("blend-standings, covered:", W.paired_boot(c.blend, c.standings)[:2])
    for lo, hi, name in ((2020, 2022, "inner"), (2023, 2025, "outer")):
        z = D[(D.season >= lo) & (D.season <= hi)]
        print(name, "full season: ens %.4f blend %.4f standings %.4f (n=%d)" % (z.ens.mean(), z.blend.mean(), z.standings.mean(), len(z)))
    z = D[D.season >= 2023]; print("outer blend-ens full:", W.paired_boot(z.blend, z.ens)[:2], "blend-standings:", W.paired_boot(z.blend, z.standings)[:2])


if __name__ == "__main__":
    main()
