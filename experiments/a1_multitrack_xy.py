"""E113: repeatability of prior-race X/Y speed signatures across three circuits."""

from itertools import permutations, product

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from experiments.a1_race_xy_pilot import profile
from f1.data import DATA


PART = DATA / "parts/race_xy_profiles_2023_2024.csv"
SESSIONS = {
    "Bahrain": {2023: 7953, 2024: 9472},
    "Austria": {2023: 9118, 2024: 9550},
    "Monza": {2023: 9157, 2024: 9590},
}
DRIVERS = {1: "VER", 11: "PER", 16: "LEC", 55: "SAI", 44: "HAM", 63: "RUS"}
TEAMMATES = [(1, 11), (16, 55), (44, 63)]


def extract():
    if PART.exists():
        cached = pd.read_csv(PART)
        complete = cached.groupby(["circuit", "season", "driver", "half"])["bin"].nunique()
        if len(complete) == len(SESSIONS) * 2 * len(DRIVERS) * 2 and complete.min() >= 40:
            return cached
    rows = []
    for circuit, years in SESSIONS.items():
        for year, session in years.items():
            samples = {driver: profile(session, driver) for driver in DRIVERS}
            ref = samples[1][samples[1].lap == 1].sort_values("date")
            xy = ref[["x", "y"]].to_numpy()
            dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            bins = np.minimum((dist / dist[-1] * 50).astype(int), 49)
            tree = cKDTree(xy)
            for driver, s in samples.items():
                nearest_dist, nearest = tree.query(s[["x", "y"]].to_numpy())
                s = s.copy()
                s["bin"] = bins[nearest]
                s = s[nearest_dist < 100]
                s["half"] = np.where(s.lap <= 2, 0, 1)
                g = s.groupby(["half", "bin"]).speed.median()
                for (half, bin_), speed in g.items():
                    rows.append((circuit, year, driver, half, bin_, speed))
            print(circuit, year, "done", flush=True)
            pd.DataFrame(rows, columns=["circuit", "season", "driver", "half", "bin", "speed"]).to_csv(PART, index=False)
    return pd.DataFrame(rows, columns=["circuit", "season", "driver", "half", "bin", "speed"])


def main():
    d = extract()
    field = d.groupby(["circuit", "season", "half", "bin"]).speed.median().rename("field_speed")
    d = d.join(field, on=["circuit", "season", "half", "bin"])
    d["residual"] = d.speed - d.field_speed
    pivot = d.pivot_table(index=["circuit", "season", "driver", "bin"], columns="half", values=["speed", "residual"])
    split = []
    for (circuit, season, driver), g in pivot.groupby(level=[0, 1, 2]):
        split.append((circuit, season, driver, g["speed"][0].corr(g["speed"][1]),
                      g["residual"][0].corr(g["residual"][1]), len(g.dropna())))
    split = pd.DataFrame(split, columns=["circuit", "season", "driver", "raw_corr", "resid_corr", "bins"])
    print("split-half by circuit", split.groupby("circuit")[["raw_corr", "resid_corr", "bins"]].agg(["mean", "min", "max"]).round(3).to_string())
    cross = []
    ids = list(DRIVERS)
    for circuit in SESSIONS:
        profiles = {}
        for year in (2023, 2024):
            for driver in ids:
                s = d[(d.circuit == circuit) & (d.season == year) & (d.driver == driver)]
                profiles[year, driver] = s.groupby("bin").residual.mean().reindex(range(50))
        corr = {(a, b): profiles[2023, a].corr(profiles[2024, b]) for a in ids for b in ids}
        actual = np.mean([corr[d_, d_] for d_ in ids])
        all_null = [np.mean([corr[d_, p] for d_, p in zip(ids, perm)]) for perm in permutations(ids)]
        teammate_null = []
        for flips in product((0, 1), repeat=len(TEAMMATES)):
            mapping = {d_: d_ for d_ in ids}
            for (a, b), flip in zip(TEAMMATES, flips):
                if flip:
                    mapping[a], mapping[b] = b, a
            teammate_null.append(np.mean([corr[d_, mapping[d_]] for d_ in ids]))
        cross.extend((circuit, d_, corr[d_, d_]) for d_ in ids)
        print(circuit, "cross-year residual identity mean", round(actual, 4),
              "all-driver permutation rank", sum(v >= actual for v in all_null), "of", len(all_null),
              "within-team-swap rank", sum(v >= actual for v in teammate_null), "of", len(teammate_null))
    cross = pd.DataFrame(cross, columns=["circuit", "driver", "corr"])
    print("cross-year residual summary", cross.groupby("circuit")["corr"].agg(["mean", "median", "min", "max"]).round(4).to_string())
    print("overall", cross["corr"].agg(["mean", "median", "min", "max"]).round(4).to_string(),
          "positive", int((cross["corr"] > 0).sum()), "of", len(cross))


if __name__ == "__main__":
    main()
