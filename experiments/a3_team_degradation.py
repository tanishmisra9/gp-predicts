"""E145: does prior team-relative race tyre degradation transfer?"""

import warnings

import fastf1
import numpy as np
import pandas as pd

import f1.ff1  # initialize the project's existing FastF1 cache
import f1.walkforward as W
from experiments.b6_adaptation_gate import LINEAGE
from f1.data import DATA, load_results


PART = DATA / "parts/team_degradation_2018_2025.csv"


def team_slopes(laps, entrants):
    l = laps[(laps.LapNumber > 1) & laps.LapTime.notna() & laps.PitInTime.isna()
             & laps.PitOutTime.isna() & laps.TrackStatus.astype(str).eq("1")
             & laps.IsAccurate.fillna(False) & laps.TyreLife.notna() & laps.Compound.notna()].copy()
    l["t"] = l.LapTime.dt.total_seconds()
    l = l[l.t < 1.07 * l.groupby("Driver").t.transform("median")]
    l["residual"] = l.t - l.groupby("LapNumber").t.transform("median")
    rows = []
    for (code, stint, compound), g in l.groupby(["Driver", "Stint", "Compound"]):
        if len(g) < 8 or g.TyreLife.nunique() < 6:
            continue
        slope = np.polyfit(g.TyreLife, g.residual, 1)[0]
        if np.isfinite(slope) and abs(slope) <= 1:
            rows.append((code, compound, slope, len(g)))
    if not rows:
        return pd.DataFrame(columns=["team", "relative_deg", "n_drivers"])
    s = pd.DataFrame(rows, columns=["code", "compound", "slope", "n"])
    s = s.groupby(["code", "compound"]).apply(
        lambda g: pd.Series({"slope": np.average(g.slope, weights=g.n)}),
        include_groups=False).reset_index()
    s["relative"] = s.slope - s.groupby("compound").slope.transform("median")
    driver = s.groupby("code", as_index=False).relative.mean()
    driver = driver.merge(entrants[["code", "team"]], on="code", validate="one_to_one")
    driver["team"] = driver.team.replace(LINEAGE)
    return driver.groupby("team", as_index=False).agg(
        relative_deg=("relative", "mean"), n_drivers=("code", "nunique"))


def extract(results):
    fastf1.Cache.enable_cache(str(DATA / "fastf1_cache"))
    fastf1.Cache.offline_mode(True)
    rows = pd.read_csv(PART).to_dict("records") if PART.exists() else []
    done = {(x["season"], x["round"]) for x in rows}
    for (year, round_), entrants in results.groupby(["season", "round"], sort=True):
        if (year, round_) in done:
            continue
        try:
            session = fastf1.get_session(int(year), int(round_), "R")
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            p = team_slopes(session.laps, entrants)
            if len(p):
                rows.extend(p.assign(season=year, round=round_).to_dict("records"))
                pd.DataFrame(rows).to_csv(PART, index=False)
            done.add((year, round_))
            print(year, round_, "teams", len(p), flush=True)
        except Exception as exc:
            print(year, round_, "unavailable", type(exc).__name__, str(exc)[:100], flush=True)
    return pd.DataFrame(rows)


def prior_team(results, source):
    events = results[["season", "round", "date"]].drop_duplicates().sort_values(["date", "season", "round"])
    events["k"] = np.arange(len(events))
    s = source.merge(events[["season", "round", "k"]], on=["season", "round"], validate="many_to_one")
    teams = results[["season", "round", "team"]].drop_duplicates().copy()
    teams["team"] = teams.team.replace(LINEAGE)
    teams = teams.merge(events[["season", "round", "k"]], on=["season", "round"], validate="many_to_one")
    output = []
    for k, cur in teams.groupby("k", sort=True):
        past = s[s.k < k].copy()
        past["w"] = 0.5 ** ((k - past.k) / 10)
        weighted = (past.relative_deg * past.w).groupby(past.team).sum()
        weights = past.w.groupby(past.team).sum()
        f = cur[["season", "round", "team"]].copy()
        f["prior_deg"] = f.team.map(weighted / (weights + 10)).fillna(0)
        output.append(f)
    return pd.concat(output, ignore_index=True)


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    results = load_results().query("season <= 2025")
    source = extract(results)
    print("source", source.groupby("season").agg(races=("round", "nunique"),
                                              teams=("team", "size"),
                                              min_drivers=("n_drivers", "min")).to_string())
    forecast = prior_team(results, source)
    target = source.merge(forecast, on=["season", "round", "team"], validate="one_to_one")
    target = target[target.season.between(2020, 2025)].copy()
    target["candidate_mse"] = (target.prior_deg - target.relative_deg) ** 2
    target["zero_mse"] = target.relative_deg ** 2
    print("target coverage", target.groupby("season").agg(races=("round", "nunique"),
                                                          teams=("team", "size")).to_string())
    print("MSE by season", target.groupby("season")[["candidate_mse", "zero_mse"]].mean().to_string())
    for label, years in (("inner", range(2020, 2023)), ("outer", range(2023, 2026))):
        g = target[target.season.isin(years)]
        races = g.groupby(["season", "round"])[["candidate_mse", "zero_mse"]].mean()
        print(label, "team-races", len(g), "races", len(races),
              "candidate/zero MSE", g.candidate_mse.mean(), g.zero_mse.mean(),
              "race-paired diff", W.paired_boot(races.candidate_mse, races.zero_mse)[:2],
              "forecast/outcome corr", g.prior_deg.corr(g.relative_deg))
        rng = np.random.default_rng(145)
        actual = g.relative_deg.to_numpy()
        forecasted = g.prior_deg.to_numpy()
        groups = [np.array(idx) for idx in g.reset_index(drop=True).groupby(["season", "round"]).indices.values()]
        null = np.empty(5000)
        for i in range(len(null)):
            shuffled = actual.copy()
            for ix in groups:
                shuffled[ix] = rng.permutation(shuffled[ix])
            null[i] = np.corrcoef(forecasted, shuffled)[0, 1]
        print(label, "within-race shuffle null mean", null.mean(),
              "one-sided p", (1 + (null >= g.prior_deg.corr(g.relative_deg)).sum()) / (len(null) + 1))


if __name__ == "__main__":
    main()
