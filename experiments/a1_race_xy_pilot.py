"""Small race-trim X/Y profile reliability pilot; diagnostic, not a podium model."""

from itertools import permutations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from f1.laps import get


SESSIONS = {2023: 7953, 2024: 9472}  # Bahrain races
DRIVERS = {1: "VER", 11: "PER", 16: "LEC", 55: "SAI"}


def profile(session, driver):
    laps = pd.DataFrame(get("laps", session_key=session, driver_number=driver))
    # This fixed early stint avoids the first pit stops in both races.
    laps = laps[laps.lap_number.between(6, 10) & ~laps.is_pit_out_lap.astype(bool)].sort_values("lap_number")
    starts = pd.to_datetime(laps.date_start, utc=True)
    start = (starts.min() - pd.Timedelta(seconds=2)).isoformat()
    end = (starts.iloc[-1] + pd.Timedelta(seconds=float(laps.lap_duration.iloc[-1]))).isoformat()
    params = {"session_key": session, "driver_number": driver, "date>": start, "date<": end}
    loc = pd.DataFrame(get("location", **params))
    car = pd.DataFrame(get("car_data", **params))
    for frame in (loc, car):
        frame["date"] = pd.to_datetime(frame.date, utc=True)
    samples = pd.merge_asof(car.sort_values("date"), loc.sort_values("date"),
                            on="date", by=["session_key", "driver_number", "meeting_key"],
                            tolerance=pd.Timedelta(milliseconds=400), direction="nearest")
    samples = samples.dropna(subset=["x", "y"])
    samples["lap"] = np.searchsorted(starts.to_numpy(), samples.date.to_numpy(), side="right")
    samples = samples[samples.lap.between(1, len(laps))]
    return samples


def main():
    all_samples = {(year, driver): profile(session, driver)
                   for year, session in SESSIONS.items() for driver in DRIVERS}
    outputs = {}
    for year in SESSIONS:
        # Reference is one earlier completed race lap. Each year's coordinate
        # system is registered independently at the timing line.
        ref = all_samples[year, 1]
        ref = ref[ref.lap == 1].sort_values("date")
        xy = ref[["x", "y"]].to_numpy()
        dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        bins = np.minimum((dist / dist[-1] * 50).astype(int), 49)
        tree = cKDTree(xy)
        for driver in DRIVERS:
            s = all_samples[year, driver].copy()
            nearest_dist, nearest = tree.query(s[["x", "y"]].to_numpy())
            s["bin"] = bins[nearest]
            s = s[nearest_dist < 100]
            s["half"] = np.where(s.lap <= 2, 0, 1)
            outputs[year, driver] = s.groupby(["half", "bin"]).speed.median().unstack(0).reindex(range(50))
            print(year, DRIVERS[driver], "samples", len(s), "bins", outputs[year, driver].notna().all(axis=1).sum())
    # Remove the common circuit speed curve. Raw profile correlation mostly
    # measures track geometry, not a persistent car-specific response.
    residual = {}
    for year in SESSIONS:
        for half in (0, 1):
            field = pd.concat([outputs[year, d][half] for d in DRIVERS], axis=1).median(axis=1)
            for driver in DRIVERS:
                residual[year, driver, half] = outputs[year, driver][half] - field
        for driver in DRIVERS:
            p = outputs[year, driver]
            print("split_half_raw", year, DRIVERS[driver], round(p[0].corr(p[1]), 4))
            print("split_half_residual", year, DRIVERS[driver],
                  round(residual[year, driver, 0].corr(residual[year, driver, 1]), 4))
    for driver in DRIVERS:
        a = outputs[2023, driver].mean(axis=1)
        b = outputs[2024, driver].mean(axis=1)
        ra = (residual[2023, driver, 0] + residual[2023, driver, 1]) / 2
        rb = (residual[2024, driver, 0] + residual[2024, driver, 1]) / 2
        print("cross_year_raw", DRIVERS[driver], round(a.corr(b), 4))
        print("cross_year_residual", DRIVERS[driver], round(ra.corr(rb), 4))
    prior = {d: (residual[2023, d, 0] + residual[2023, d, 1]) / 2 for d in DRIVERS}
    later = {d: (residual[2024, d, 0] + residual[2024, d, 1]) / 2 for d in DRIVERS}
    ids = list(DRIVERS)
    actual = np.mean([prior[d].corr(later[d]) for d in ids])
    null = [np.mean([prior[d].corr(later[p]) for d, p in zip(ids, perm)])
            for perm in permutations(ids)]
    print("cross_year_residual_mean", round(actual, 4), "permutation_rank",
          sum(x >= actual for x in null), "of", len(null))


if __name__ == "__main__":
    main()
