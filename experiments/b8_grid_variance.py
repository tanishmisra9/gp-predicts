"""E133: does past driver-specific grid uncertainty forecast future setbacks?"""

import numpy as np
import pandas as pd
from scipy.stats import norm

import f1.walkforward as W
from experiments.b8_grid_feature import add_prior_grid
from experiments.common import table


def prior_variance(X):
    """Past-only grid-error variance for each standings top-six contender."""
    d = X.copy()
    d["stand_score"] = -d.d_season_pts.fillna(0) * 100 + d.d10_order.fillna(11) * 1e-3
    d = d.sort_values(["k", "stand_score", "driver"])
    d["stand_rank"] = d.groupby("k").cumcount() + 1
    prior = d[(d.stand_rank <= 6) & d.prior_grid.notna()].copy()
    prior["error2"] = (prior.gridn - prior.prior_grid) ** 2
    rows = []
    for k, g in d[d.season.between(2020, 2025) & d.stand_rank.le(6)].groupby("k"):
        past = prior[prior.k < k]
        weights = 0.5 ** ((k - past.k.to_numpy()) / 20)
        global_var = np.average(past.error2, weights=weights) if len(past) else 16.0
        for row in g.itertuples():
            own = past[past.driver == row.driver]
            own_weights = 0.5 ** ((k - own.k.to_numpy()) / 20)
            own_var = (8 * global_var + np.dot(own_weights, own.error2)) / (8 + own_weights.sum())
            rows.append(dict(season=row.season, round=row.round, k=k, driver=row.driver,
                             code=row.code, prior_grid=row.prior_grid, gridn=row.gridn,
                             global_var=max(global_var, 1), driver_var=max(own_var, 1),
                             prior_own_weight=own_weights.sum()))
    return pd.DataFrame(rows)


def main():
    X = table()
    X = add_prior_grid(X[X.season <= 2025].copy())
    d = prior_variance(X)
    assert d.groupby(["season", "round"]).size().eq(6).all()
    assert d.prior_grid.notna().all()
    d["setback"] = d.gridn.gt(6).astype(float)
    for name in ("global", "driver"):
        d[name + "_p"] = norm.sf((6.5 - d.prior_grid) / np.sqrt(d[name + "_var"]))
        d[name + "_brier"] = (d[name + "_p"] - d.setback) ** 2
    d["split"] = np.where(d.season <= 2022, "inner", "outer")
    print("strict podium unchanged: standings 0.6066 / 0.5905, frozen ens 0.6284 / 0.5857,"
          " shuffled-label podium null 0.1749 / 0.1857")
    print("top-six grid setback rate", d.groupby("split").setback.mean().round(4).to_dict())
    print("top-six Brier", d.groupby("split")[["global_brier", "driver_brier"]].mean().round(6).to_string())
    race = d.groupby(["split", "season", "round"])[["global_brier", "driver_brier"]].mean()
    for split in ("inner", "outer"):
        g = race.loc[split]
        print(split, "driver minus global race-paired Brier", W.paired_boot(g.driver_brier, g.global_brier)[:2],
              "races", len(g))
        groups = [v for _, v in d[d.split == split].groupby(["season", "round"])]
        labels = np.stack([v.setback.to_numpy() for v in groups])
        global_p = np.stack([v.global_p.to_numpy() for v in groups])
        driver_p = np.stack([v.driver_p.to_numpy() for v in groups])
        observed = ((driver_p - labels) ** 2 - (global_p - labels) ** 2).mean()
        rng = np.random.default_rng(133)
        shuffled = []
        for _ in range(2000):
            y = np.stack([rng.permutation(row) for row in labels])
            shuffled.append(((driver_p - y) ** 2 - (global_p - y) ** 2).mean())
        shuffled = np.asarray(shuffled)
        print(split, "within-race grid-label shuffle mean", round(shuffled.mean(), 6),
              "one-sided p", round((1 + (shuffled <= observed).sum()) / (len(shuffled) + 1), 4))
    print("driver prior weight", d.groupby("split").prior_own_weight.agg(["median", "min", "max"]).round(3).to_string())


if __name__ == "__main__":
    main()
