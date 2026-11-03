"""E125: can past team behaviour forecast relative first-stop timing?"""

import glob

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.common import table
from f1.data import DATA, load_results


def source_table(X, pits, results):
    """Team first-stop fraction minus that race's field median, for full-distance cars."""
    keys = X[["season", "round", "code", "team", "k"]]
    races = results[["season", "round", "code", "laps"]].copy()
    races["race_laps"] = races.groupby(["season", "round"]).laps.transform("max")
    z = pits[pits.season <= 2025].merge(keys, on=["season", "round", "code"], validate="one_to_one")
    z = z.merge(races, on=["season", "round", "code"], validate="one_to_one")
    z = z[z.first_stop.notna() & (z.laps >= 0.9 * z.race_laps)].copy()
    z["fraction"] = z.first_stop / z.race_laps
    z["resid"] = z.fraction - z.groupby(["season", "round"]).fraction.transform("median")
    return z.groupby(["season", "round", "team", "k"], as_index=False).agg(
        resid=("resid", "mean"), drivers=("code", "size"))


def past_first_stop(X, source):
    """Ten-race half-life, ten-team-observation zero prior; strictly earlier races."""
    out = []
    for k, cur in X.groupby("k", sort=True):
        past = source[source.k < k].copy()
        past["w"] = past.drivers * 0.5 ** ((k - past.k) / 10)
        num = (past.resid * past.w).groupby(past.team).sum()
        den = past.w.groupby(past.team).sum()
        rate = num / (den + 10)
        f = cur[["season", "round", "team"]].drop_duplicates().copy()
        f["forecast"] = f.team.map(rate).fillna(0)
        out.append(f)
    return pd.concat(out, ignore_index=True)


def main():
    X = table(); X = X[X.season <= 2025].copy()
    pits = pd.concat([pd.read_csv(f) for f in glob.glob(str(DATA / "parts/pits_*.csv"))], ignore_index=True)
    source = source_table(X, pits, load_results())
    forecast = past_first_stop(X, source)
    d = source.merge(forecast, on=["season", "round", "team"], validate="one_to_one")
    d = d[d.season.between(2020, 2025)].copy()
    d["split"] = np.where(d.season <= 2022, "inner", "outer")
    d["model_sq"] = (d.resid - d.forecast) ** 2
    d["zero_sq"] = d.resid ** 2
    print("source", source.groupby(["season", "round"]).ngroups, "races", len(source), "team-races")
    for split, g in d.groupby("split"):
        race = g.groupby(["season", "round"])[["model_sq", "zero_sq"]].mean()
        diff, ci, _ = W.paired_boot(race.model_sq, race.zero_sq)
        print(split, "coverage", len(g), "team-races", len(race), "races",
              "rho", round(g.forecast.corr(g.resid, method="spearman"), 4),
              "model MSE", round(g.model_sq.mean(), 6), "zero MSE", round(g.zero_sq.mean(), 6),
              "paired MSE difference", round(diff, 6), np.round(ci, 6).tolist())
    for season, g in d.groupby("season"):
        print(season, "rho", round(g.forecast.corr(g.resid, method="spearman"), 4),
              "MSE difference", round((g.model_sq - g.zero_sq).mean(), 6))
    outer = d[d.split == "outer"]
    rng = np.random.default_rng(125)
    observed = outer.forecast.corr(outer.resid, method="spearman")
    null = np.array([outer.forecast.corr(outer.groupby(["season", "round"]).resid.transform(
        lambda x: rng.permutation(x)), method="spearman") for _ in range(2000)])
    print("outer within-race label-shuffle rho mean", round(null.mean(), 4),
          "one-sided p", round((1 + (null >= observed).sum()) / (len(null) + 1), 4))
    print("strict podium reference: standings 0.6066 inner / 0.5905 outer; frozen ens 0.6284 / 0.5857; shuffled-label logit null 0.1749 / 0.1857")


if __name__ == "__main__":
    main()
