"""E128: unchanged E127 multi-stint profile on Bahrain, Austria, and Monza."""

from itertools import permutations, product

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from experiments.a1_multistint_xy import PART as BAHRAIN_PART, one_driver
from experiments.a1_multitrack_xy import DRIVERS, SESSIONS, TEAMMATES
from experiments.a1_race_xy_pilot import profile
from f1.data import DATA
from f1.laps import get


PART = DATA / "parts/race_xy_multistint_3circuits_2023_2024.csv"


def extract():
    if PART.exists():
        rows = pd.read_csv(PART).to_dict("records")
    else:
        rows = pd.read_csv(BAHRAIN_PART).assign(circuit="Bahrain").to_dict("records")
    done = {(r["circuit"], r["season"], r["driver"]) for r in rows}
    for circuit, years in SESSIONS.items():
        for season, session in years.items():
            if all((circuit, season, driver) in done for driver in DRIVERS):
                continue
            all_laps = pd.DataFrame(get("laps", session_key=session))
            ref = profile(session, 1)
            ref = ref[ref.lap == 1].sort_values("date")
            xy = ref[["x", "y"]].to_numpy()
            distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            bins = np.minimum((distance / distance[-1] * 50).astype(int), 49)
            tree = cKDTree(xy)
            for driver in DRIVERS:
                if (circuit, season, driver) in done:
                    continue
                try:
                    p = one_driver(session, driver, all_laps, tree, bins)
                    rows.extend(p.assign(circuit=circuit, season=season, driver=driver).to_dict("records"))
                    pd.DataFrame(rows).to_csv(PART, index=False)
                    print(circuit, season, DRIVERS[driver], "laps", p.clean_laps.iloc[0],
                          "samples", p.samples.sum(), "bins", p.groupby("half").bin.nunique().to_dict(), flush=True)
                except Exception as exc:
                    print(circuit, season, DRIVERS[driver], "unavailable", type(exc).__name__, str(exc)[:120], flush=True)
    return pd.DataFrame(rows)


def main():
    d = extract()
    complete = d.groupby(["circuit", "season", "driver"]).half.nunique()
    if len(complete) != len(SESSIONS) * 2 * len(DRIVERS) or complete.min() != 2:
        raise RuntimeError(f"incomplete driver-race coverage: {len(complete)} of {len(SESSIONS) * 2 * len(DRIVERS)}")
    field = d.groupby(["circuit", "season", "half", "bin"]).speed.median().rename("field")
    d = d.join(field, on=["circuit", "season", "half", "bin"])
    d["resid"] = d.speed - d.field
    p = d.pivot_table(index=["circuit", "season", "driver", "bin"], columns="half", values="resid")
    split = p.groupby(level=[0, 1, 2]).apply(lambda g: g[0].corr(g[1]))
    print("within-race split-half residual mean by circuit", split.groupby(level="circuit").mean().round(4).to_dict())
    ids = list(DRIVERS)
    corr = {}
    for circuit in SESSIONS:
        profiles = {(year, driver): d[(d.circuit == circuit) & (d.season == year) & (d.driver == driver)]
                    .groupby("bin").resid.mean().reindex(range(50)) for year in (2023, 2024) for driver in ids}
        corr[circuit] = {(a, b): profiles[2023, a].corr(profiles[2024, b]) for a in ids for b in ids}
        actual = np.mean([corr[circuit][driver, driver] for driver in ids])
        all_null = [np.mean([corr[circuit][driver, p] for driver, p in zip(ids, perm)])
                    for perm in permutations(ids)]
        teammate_null = []
        for flips in product((0, 1), repeat=len(TEAMMATES)):
            mapping = {driver: driver for driver in ids}
            for (a, b), flip in zip(TEAMMATES, flips):
                if flip:
                    mapping[a], mapping[b] = b, a
            teammate_null.append(np.mean([corr[circuit][driver, mapping[driver]] for driver in ids]))
        print(circuit, "cross-year mean", round(actual, 4),
              "all-driver rank", sum(v >= actual for v in all_null), "of", len(all_null),
              "teammate-swap rank", sum(v >= actual for v in teammate_null), "of", len(teammate_null))
    actual = np.mean([corr[circuit][driver, driver] for circuit in SESSIONS for driver in ids])
    all_null = [np.mean([corr[circuit][driver, p] for circuit in SESSIONS for driver, p in zip(ids, perm)])
                for perm in permutations(ids)]
    teammate_null = []
    for flips in product((0, 1), repeat=len(TEAMMATES)):
        mapping = {driver: driver for driver in ids}
        for (a, b), flip in zip(TEAMMATES, flips):
            if flip:
                mapping[a], mapping[b] = b, a
        teammate_null.append(np.mean([corr[circuit][driver, mapping[driver]] for circuit in SESSIONS for driver in ids]))
    print("combined cross-year mean", round(actual, 4), "all-driver rank",
          sum(v >= actual for v in all_null), "of", len(all_null),
          "teammate-swap rank", sum(v >= actual for v in teammate_null), "of", len(teammate_null))
    print("coverage", d.groupby(["circuit", "season", "driver"]).agg(
        clean_laps=("clean_laps", "first"), bins=("bin", "nunique")).groupby("circuit").agg(
        driver_races=("clean_laps", "size"), min_laps=("clean_laps", "min"), max_laps=("clean_laps", "max"),
        min_bins=("bins", "min")).to_string())


if __name__ == "__main__":
    main()
