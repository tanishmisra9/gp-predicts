"""E124: does pre-season testing mileage forecast early-season attrition?"""

import fastf1
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.common import table
from f1.data import DATA, load_results


PART = DATA / "parts/testing_mileage_2023_2025.csv"


def extract():
    """Count recorded testing laps by driver/day; fail if any expected day is absent."""
    if PART.exists():
        return pd.read_csv(PART)
    fastf1.Cache.enable_cache(str(DATA / "fastf1_cache"))
    fastf1.Cache.offline_mode(True)
    rows = []
    for season in (2023, 2024, 2025):
        for day in (1, 2, 3):
            session = fastf1.get_testing_session(season, 1, day)
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            laps = session.laps.groupby("Driver").size()
            rows.extend({"season": season, "day": day, "code": code, "laps": count}
                        for code, count in laps.items())
    out = pd.DataFrame(rows)
    out.to_csv(PART, index=False)
    return out


def team_mileage(entries, source):
    """Map pre-season driver codes to each season's round-one team entry list."""
    first = entries[entries["round"] == 1][["season", "code", "team"]].drop_duplicates()
    matched = source.merge(first, on=["season", "code"], how="inner", validate="many_to_one")
    return matched.groupby(["season", "team"], as_index=False).laps.sum()


def main():
    source = extract()
    X = table(); X = X[X.season.between(2023, 2025)].copy()
    mileage = team_mileage(X, source)
    results = load_results()
    results = results[results.season.between(2023, 2025) & results["round"].between(1, 3)].copy()
    results["dnf"] = (~(results.status.isin(["Finished", "Lapped"]) |
                        results.status.str.startswith("+"))).astype(int)
    outcome = results.groupby(["season", "team"], as_index=False).agg(
        dnf=("dnf", "mean"), entrants=("code", "size"))
    prior = X[X["round"] == 1].groupby(["season", "team"], as_index=False).t10_dnf.mean()
    d = outcome.merge(mileage, on=["season", "team"], validate="one_to_one")
    d = d.merge(prior, on=["season", "team"], validate="one_to_one")
    d["t10_dnf"] = d.t10_dnf.fillna(d.groupby("season").t10_dnf.transform("mean"))
    d["mileage_risk"] = -d.laps
    print("source days", source.groupby("season").day.nunique().to_dict(),
          "source codes", source.groupby("season").code.nunique().to_dict())
    print("covered team-years", len(d), "of", len(outcome), "driver-races", d.entrants.sum(),
          "dnfs", round((d.dnf * d.entrants).sum()))
    print("mileage laps", d.groupby("season").laps.agg(["min", "median", "max"]).to_string())
    for season, g in d.groupby("season"):
        print(season, "rho risk vs early DNF: mileage", round(spearmanr(g.mileage_risk, g.dnf).statistic, 4),
              "prior t10 DNF", round(spearmanr(g.t10_dnf, g.dnf).statistic, 4))
    season_rho = lambda y: np.mean([spearmanr(g[y], g.dnf).statistic for _, g in d.groupby("season")])
    print("mean within-season rho mileage", round(season_rho("mileage_risk"), 4),
          "prior", round(season_rho("t10_dnf"), 4))
    rng = np.random.default_rng(124)
    observed = season_rho("mileage_risk")
    null = np.array([np.mean([spearmanr(g.mileage_risk, rng.permutation(g.dnf)).statistic
                              for _, g in d.groupby("season")]) for _ in range(5000)])
    print("within-season label-shuffle null mean rho", round(null.mean(), 4),
          "one-sided p", round((1 + (null >= observed).sum()) / (len(null) + 1), 4))
    print("strict podium reference: standings 0.6066 inner / 0.5905 outer; frozen ens 0.6284 / 0.5857; shuffled-label logit null 0.1749 / 0.1857")


if __name__ == "__main__":
    main()
