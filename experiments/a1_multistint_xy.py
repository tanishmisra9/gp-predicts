"""E127: multi-stint X/Y race-speed profile versus E113's five-lap profile."""

from itertools import permutations, product

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from experiments.a1_multitrack_xy import DRIVERS, TEAMMATES
from experiments.a1_race_xy_pilot import profile
from f1.data import DATA
from f1.laps import _mark_clean, get


PART = DATA / "parts/race_xy_multistint_bahrain_2023_2024.csv"
SESSIONS = {2023: 7953, 2024: 9472}


def one_driver(session, driver, all_laps, tree, bins):
    """Return two independent stint-spanning halves of green-lap position profiles."""
    clean = _mark_clean(all_laps)
    clean = clean[(clean.driver_number == driver) & clean.date_start.notna()
                  & clean.lap_duration.notna()].sort_values("lap_number").copy()
    if len(clean) < 16:
        raise ValueError(f"only {len(clean)} clean laps for driver {driver}")
    clean["half"] = (np.arange(len(clean)) >= len(clean) // 2).astype(int)
    full = all_laps[(all_laps.driver_number == driver) & all_laps.date_start.notna()]
    full = full.sort_values("date_start")
    starts = pd.to_datetime(full.date_start, utc=True, format="ISO8601")
    first = pd.to_datetime(clean.date_start.iloc[0], utc=True) - pd.Timedelta(seconds=1)
    last = pd.to_datetime(clean.date_start.iloc[-1], utc=True) + pd.Timedelta(seconds=float(clean.lap_duration.iloc[-1]))
    params = {"session_key": session, "driver_number": driver,
              "date>": first.isoformat(), "date<": last.isoformat()}
    loc = pd.DataFrame(get("location", **params))
    car = pd.DataFrame(get("car_data", **params))
    if loc.empty or car.empty:
        raise ValueError(f"no position/car samples for driver {driver}")
    for frame in (loc, car):
        frame["date"] = pd.to_datetime(frame.date, utc=True, format="ISO8601")
    samples = pd.merge_asof(car.sort_values("date"), loc.sort_values("date"),
                            on="date", by=["session_key", "driver_number", "meeting_key"],
                            tolerance=pd.Timedelta(milliseconds=400), direction="nearest")
    samples = samples.dropna(subset=["x", "y"]).copy()
    idx = np.searchsorted(starts.to_numpy(), samples.date.to_numpy(), side="right") - 1
    samples = samples[idx >= 0].copy(); idx = idx[idx >= 0]
    samples["lap_number"] = full.lap_number.to_numpy()[idx]
    samples = samples.merge(clean[["lap_number", "date_start", "lap_duration", "half"]],
                            on="lap_number", how="inner", validate="many_to_one")
    end = pd.to_datetime(samples.date_start, utc=True, format="ISO8601") + pd.to_timedelta(samples.lap_duration, unit="s")
    samples = samples[samples.date <= end].copy()
    distance, nearest = tree.query(samples[["x", "y"]].to_numpy())
    samples = samples[distance < 100].copy()
    samples["bin"] = bins[nearest[distance < 100]]
    g = samples.groupby(["half", "bin"]).speed.agg(["median", "size"])
    if len(g.groupby(level="half")) != 2 or g.groupby(level="half").size().min() < 40:
        raise ValueError(f"incomplete track coverage for driver {driver}")
    out = g.reset_index().rename(columns={"median": "speed", "size": "samples"})
    out["clean_laps"] = len(clean)
    return out


def extract():
    rows = pd.read_csv(PART).to_dict("records") if PART.exists() else []
    done = {(r["season"], r["driver"]) for r in rows}
    for season, session in SESSIONS.items():
        all_laps = pd.DataFrame(get("laps", session_key=session))
        ref = profile(session, 1)
        ref = ref[ref.lap == 1].sort_values("date")
        xy = ref[["x", "y"]].to_numpy()
        distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        bins = np.minimum((distance / distance[-1] * 50).astype(int), 49)
        tree = cKDTree(xy)
        for driver in DRIVERS:
            if (season, driver) in done:
                continue
            try:
                p = one_driver(session, driver, all_laps, tree, bins)
                rows.extend(p.assign(season=season, driver=driver).to_dict("records"))
                pd.DataFrame(rows).to_csv(PART, index=False)
                print(season, DRIVERS[driver], "laps", p.clean_laps.iloc[0],
                      "samples", p.samples.sum(), "bins", p.groupby("half").bin.nunique().to_dict(), flush=True)
            except Exception as exc:
                print(season, DRIVERS[driver], "unavailable", type(exc).__name__, str(exc)[:120], flush=True)
    return pd.DataFrame(rows)


def main():
    d = extract()
    field = d.groupby(["season", "half", "bin"]).speed.median().rename("field")
    d = d.join(field, on=["season", "half", "bin"])
    d["resid"] = d.speed - d.field
    p = d.pivot_table(index=["season", "driver", "bin"], columns="half", values=["speed", "resid"])
    split = [g["resid"][0].corr(g["resid"][1]) for _, g in p.groupby(level=[0, 1])]
    print("within-race split-half residual mean", round(np.nanmean(split), 4),
          "median", round(np.nanmedian(split), 4), "n", len(split))
    ids = list(DRIVERS)
    profiles = {(year, driver): d[(d.season == year) & (d.driver == driver)].groupby("bin").resid.mean().reindex(range(50))
                for year in SESSIONS for driver in ids}
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
    print("cross-year same-driver residual mean", round(actual, 4),
          "all-driver permutation rank", sum(v >= actual for v in all_null), "of", len(all_null),
          "teammate-swap rank", sum(v >= actual for v in teammate_null), "of", len(teammate_null))
    print("per-driver", {DRIVERS[d_]: round(corr[d_, d_], 4) for d_ in ids})


if __name__ == "__main__":
    main()
