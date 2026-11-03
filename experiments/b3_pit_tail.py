"""E121: persistence of costly pit-stop tail risk, before podium modelling."""

import glob

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.common import table
from f1.data import DATA, load_results


def source_table(X, pits):
    keys = X[["season", "round", "code", "team", "k"]]
    s = pits[pits.season <= 2025].merge(keys, on=["season", "round", "code"], validate="one_to_one")
    s["bad"] = s.pit_gap.gt(3).astype(float)
    s.loc[s.pit_gap.isna(), "bad"] = np.nan
    return s


def prior_tail(X, source):
    """Past-only team hazard with a 20-observation global prior and 10-race half-life."""
    rows = []
    for k, cur in X.groupby("k", sort=True):
        past = source[(source.k < k) & source.bad.notna()].copy()
        past["w"] = 0.5 ** ((k - past.k) / 10)
        global_p = (past.bad * past.w).sum() / past.w.sum() if len(past) else 0.08
        num = (past.bad * past.w).groupby(past.team).sum()
        den = past.w.groupby(past.team).sum()
        p = (num + 20 * global_p) / (den + 20)
        f = cur[["season", "round", "code", "team"]].copy()
        f["prior_badstop"] = f.team.map(p).fillna(global_p)
        f["global_badstop"] = global_p
        rows.append(f[["season", "round", "code", "prior_badstop", "global_badstop"]])
    return X.merge(pd.concat(rows, ignore_index=True), on=["season", "round", "code"], validate="one_to_one")


def main():
    X = table(); X = X[X.season <= 2025].copy()
    pits = pd.concat([pd.read_csv(f) for f in glob.glob(str(DATA / "parts/pits_*.csv"))], ignore_index=True)
    source = source_table(X, pits)
    X = prior_tail(X, source)
    d = X[["season", "round", "code", "prior_badstop", "global_badstop"]].merge(
        source[["season", "round", "code", "bad"]], on=["season", "round", "code"], validate="one_to_one")
    d = d[d.season.between(2020, 2025) & d.bad.notna()].copy()
    d["split"] = np.where(d.season <= 2022, "inner", "outer")
    for name in ("prior_badstop", "global_badstop"):
        p = d[name].clip(1e-6, 1 - 1e-6)
        d[name + "_brier"] = (p - d.bad) ** 2
        d[name + "_logloss"] = -(d.bad * np.log(p) + (1 - d.bad) * np.log1p(-p))
    print("source races", source.groupby(["season", "round"]).ngroups, "valid driver-races", source.bad.notna().sum(),
          "bad stops", int(source.bad.sum()))
    print("target driver-race coverage", d.groupby("split").agg(rows=("round", "size"), events=("bad", "sum")).to_string())
    print("pit hazard scores", d.groupby("split")[["prior_badstop_brier", "global_badstop_brier",
                                                   "prior_badstop_logloss", "global_badstop_logloss"]].mean().round(6).to_string())
    for split, g in d.groupby("split"):
        race = g.groupby(["season", "round"])[["prior_badstop_brier", "global_badstop_brier",
                                                "prior_badstop_logloss", "global_badstop_logloss"]].mean()
        for metric in ("brier", "logloss"):
            diff, ci, _ = W.paired_boot(race["prior_badstop_" + metric], race["global_badstop_" + metric])
            print(split, metric, "team minus global", round(diff, 6), np.round(ci, 6).tolist(), "races", len(race))
    print("strict podium reference: standings 0.6066 inner / 0.5905 outer; frozen ens 0.6284 / 0.5857; shuffled-label logit null 0.1749 / 0.1857")


if __name__ == "__main__":
    main()
