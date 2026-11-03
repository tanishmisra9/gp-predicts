"""E140: 2024 multi-stint X/Y profiles forecast 2025 team race pace at five new circuits."""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

import f1.walkforward as W
from experiments.a1_multistint_xy import one_driver
from experiments.a1_race_xy_pilot import profile
from f1.data import DATA, load_results
from f1.laps import get


SESSIONS = {"jeddah": 9480, "suzuka": 9496, "silverstone": 9558,
            "marina_bay": 9606, "yas_marina": 9662}
PART = DATA / "parts/race_xy_multistint_newtracks_2024.csv"


def extract(sessions=SESSIONS, part=PART):
    rows = pd.read_csv(part).to_dict("records") if part.exists() else []
    done = {(r["circuit"], r["code"]) for r in rows}
    for circuit, session in sessions.items():
        drivers = {int(x["driver_number"]): x["name_acronym"] for x in get("drivers", session_key=session)}
        if all((circuit, code) in done for code in drivers.values()):
            continue
        laps = pd.DataFrame(get("laps", session_key=session))
        for number in sorted(drivers):
            try:
                ref = profile(session, number)
                ref = ref[ref.lap == 1].sort_values("date")
                if len(ref) >= 2:
                    break
            except (IndexError, KeyError, ValueError):
                continue
        else:
            raise ValueError(f"no X/Y reference lap for {circuit}")
        xy = ref[["x", "y"]].to_numpy()
        distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        bins = np.minimum((distance / distance[-1] * 50).astype(int), 49)
        tree = cKDTree(xy)
        for number, code in drivers.items():
            if (circuit, code) in done:
                continue
            try:
                p = one_driver(session, number, laps, tree, bins)
                rows.extend(p.assign(circuit=circuit, season=2024, driver=number, code=code).to_dict("records"))
                pd.DataFrame(rows).to_csv(part, index=False)
                print(circuit, code, "laps", p.clean_laps.iloc[0], "bins", p.groupby("half").bin.nunique().to_dict(), flush=True)
            except Exception as exc:
                print(circuit, code, "unavailable", type(exc).__name__, str(exc)[:120], flush=True)
    return pd.DataFrame(rows)


def prior_team_speed(source, results):
    d = source[source.season == 2024].copy()
    field = d.groupby(["circuit", "half", "bin"]).speed.median().rename("field")
    d = d.join(field, on=["circuit", "half", "bin"])
    d["resid"] = d.speed - d.field
    driver = d.groupby(["circuit", "code"], as_index=False).resid.mean()
    teams = results[results.season == 2024][["circuit", "code", "team"]].drop_duplicates()
    driver = driver.merge(teams, on=["circuit", "code"], validate="one_to_one")
    return driver.groupby(["circuit", "team"], as_index=False).agg(
        prior_xy_speed=("resid", "mean"), n_profile_drivers=("code", "nunique"))


def evaluate(source, results, pace, sessions=SESSIONS):
    prior = prior_team_speed(source, results)
    race_pace = results[results.season.isin([2024, 2025])][["season", "round", "circuit", "team", "code"]].merge(
        pace[["season", "round", "code", "pace_gap"]], on=["season", "round", "code"], how="left")
    race_pace = race_pace.groupby(["season", "round", "circuit", "team"], as_index=False).pace_gap.mean()
    rows = []
    groups = []
    for circuit in sessions:
        last = race_pace[(race_pace.season == 2024) & (race_pace.circuit == circuit)][["team", "pace_gap"]]
        target = race_pace[(race_pace.season == 2025) & (race_pace.circuit == circuit)].copy()
        round_ = int(target["round"].iloc[0])
        points = results[(results.season == 2025) & (results["round"] < round_)].groupby("team").points.sum()
        g = target.merge(prior[prior.circuit == circuit], on=["circuit", "team"], how="left", validate="one_to_one")
        g = g.merge(last.rename(columns={"pace_gap": "prior_race_pace"}), on="team", validate="one_to_one")
        g["prior_points"] = g.team.map(points).fillna(0)
        g = g.dropna(subset=["pace_gap", "prior_race_pace"])
        g["prior_xy_speed"] = g.prior_xy_speed.fillna(0)
        g["n_profile_drivers"] = g.n_profile_drivers.fillna(0)
        g["standings_score"] = -g.prior_points
        g["profile_score"] = -g.prior_xy_speed
        g["prior_pace_score"] = g.prior_race_pace
        g = g.sort_values("team")
        g["blend_score"] = (g.standings_score.rank(method="first") + g.profile_score.rank(method="first")) / 2
        groups.append(g)
        rows.append(dict(circuit=circuit, teams=len(g), min_profile_drivers=int(g.n_profile_drivers.min()),
                         **{name: spearmanr(g[name + "_score"], g.pace_gap).statistic
                            for name in ("standings", "profile", "prior_pace", "blend")}))
    return pd.DataFrame(rows).set_index("circuit"), groups


def main():
    source = extract()
    results = load_results()
    results = results[results.season <= 2025]
    pace = pd.read_csv(DATA / "racepace_ff1.csv")
    r, groups = evaluate(source, results, pace)
    print("2025 team race-pace Spearman", r.round(4).to_string())
    print("mean", r.drop(columns=["teams", "min_profile_drivers"]).mean().round(4).to_dict())
    for name in ("profile", "blend", "prior_pace"):
        print(name, "minus standings paired", W.paired_boot(r[name], r.standings)[:2])
    rng = np.random.default_rng(140)
    null = np.array([np.mean([spearmanr(g.profile_score, rng.permutation(g.pace_gap)).statistic
                              for g in groups]) for _ in range(5000)])
    print("profile shuffled-pace null mean", round(null.mean(), 4),
          "one-sided p", round((1 + (null >= r.profile.mean()).sum()) / (len(null) + 1), 4))
    print("strict podium reference: standings 0.6066 inner / 0.5905 outer; frozen ens 0.6284 / 0.5857; shuffled-label logit null 0.1749 / 0.1857")


if __name__ == "__main__":
    main()
