"""E258: raw within-gear ratio episodes; no fault labels or forecast."""
import hashlib
import json
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.e190_traffic_feasibility import PILOT, cached_session
from f1.data import ROOT

EVENT_COLS = ["lap", "gear", "sign", "first_index", "last_index", "first", "last", "samples", "max_abs_log_ratio"]


def windows(laps):
    assert not laps.duplicated(["Driver", "LapNumber"]).any()
    d = laps[laps.LapNumber.gt(1) & laps.LapTime.notna() & laps.LapTime.gt(pd.Timedelta(0))
             & laps.LapStartTime.notna() & laps.Time.notna() & laps.PitInTime.isna()
             & laps.PitOutTime.isna() & laps.TrackStatus.astype(str).eq("1")].copy()
    return d


def measure(car, laps, origin):
    d = car[car.Date.notna()].drop_duplicates("Date", keep="last").sort_values("Date").reset_index(drop=True)
    n = len(d)
    summary = dict(native_samples=n, eligible=0, supported_gears=0, marked=0, episodes=0)
    empty = pd.DataFrame(columns=EVENT_COLS)
    if n < 3 or laps.empty:
        return summary, empty
    starts = laps.LapStartTime.dt.total_seconds().to_numpy()
    ends = laps.Time.dt.total_seconds().to_numpy()
    order = np.argsort(starts)
    starts, ends = starts[order], ends[order]
    lapnums = laps.LapNumber.to_numpy()[order]
    assert np.isfinite(starts).all() and np.isfinite(ends).all() and (ends > starts).all()
    assert (starts[1:] >= ends[:-1]).all(), "overlapping lap windows"
    t = (d.Date - origin).dt.total_seconds().to_numpy()
    gear = d.nGear.to_numpy(float)
    rpm = d.RPM.to_numpy(float)
    speed = d.Speed.to_numpy(float)
    valid = d.Time.notna().to_numpy() & np.isfinite(rpm) & (rpm >= 1000)
    valid &= np.isfinite(speed) & (speed >= 180) & d.Throttle.between(98, 100).to_numpy()
    valid &= d.Brake.eq(False).to_numpy() & np.isfinite(gear) & (gear >= 4) & (gear <= 8) & (gear == np.floor(gear))
    dt = np.diff(t)
    adjacent = (dt > 0) & (dt <= .5)
    centers = np.arange(1, n-1)
    keep = valid[:-2] & valid[1:-1] & valid[2:] & adjacent[:-1] & adjacent[1:]
    keep &= (gear[:-2] == gear[1:-1]) & (gear[1:-1] == gear[2:])
    window = np.searchsorted(starts, t[1:-1], side="right") - 1
    safe = np.maximum(window, 0)
    keep &= (window >= 0) & (t[:-2] >= starts[safe]) & (t[2:] < ends[safe])
    ids = centers[keep]
    if not len(ids):
        return summary, empty
    z = pd.DataFrame(dict(native_index=ids, time=t[ids], gear=gear[ids].astype(int),
                          lap=lapnums[window[keep]].astype(int), ratio=rpm[ids]/speed[ids]))
    size = z.groupby("gear").ratio.transform("size")
    baseline = z.groupby("gear").ratio.transform("median")
    z["residual"] = np.log(z.ratio / baseline)
    z["marked"] = (size >= 50) & (z.residual.abs() > np.log(1.10))
    summary.update(eligible=len(z), supported_gears=int(z.loc[size >= 50, "gear"].nunique()), marked=int(z.marked.sum()))
    marked = z[z.marked].copy()
    if marked.empty:
        return summary, empty
    marked["sign"] = np.sign(marked.residual).astype(int)
    new = marked.native_index.diff().ne(1) | marked.lap.diff().ne(0) | marked.gear.diff().ne(0) | marked.sign.diff().ne(0)
    events = []
    for _, g in marked.groupby(new.cumsum(), sort=True):
        if len(g) >= 5 and g.time.iloc[-1] - g.time.iloc[0] >= 1:
            a, b = g.iloc[0], g.iloc[-1]
            events.append(dict(lap=int(a.lap), gear=int(a.gear), sign=int(a.sign), first_index=int(a.native_index),
                               last_index=int(b.native_index), first=float(a.time), last=float(b.time), samples=len(g),
                               max_abs_log_ratio=float(g.residual.abs().max())))
    summary["episodes"] = len(events)
    return summary, pd.DataFrame(events, columns=EVENT_COLS)


def gate(support, episodes, reports):
    supported = support[(support.eligible >= 100) & (support.supported_gears >= 2)]
    counts = supported.groupby(["season", "round"]).size()
    return bool(len(reports) == 3 and all(r["available"] for r in reports)
                and all(counts.get(tuple(k), 0) >= 10 for k in PILOT)
                and episodes.groupby(["season", "round"]).ngroups >= 2 and episodes.driver.nunique() >= 3)


def main():
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True)
    rows, events, reports = [], [], []
    for season, round_ in PILOT:
        report = dict(season=season, round=round_, available=False)
        try:
            with patch("requests.adapters.HTTPAdapter.send", side_effect=RuntimeError("E258 HTTP forbidden")):
                session = cached_session(fastf1, season, round_)
                laps = pd.DataFrame(session.laps)
                cars = fastf1._api.car_data(session.api_path)
        except Exception as exc:
            report["error_type"] = type(exc).__name__
        else:
            selected = windows(laps)
            origin = max((c.Date-c.Time).max() for c in cars.values() if c.Time.notna().any()).round("ms")
            for driver, raw in laps.groupby("Driver", sort=True):
                numbers = raw.DriverNumber.dropna().unique()
                assert len(numbers) == 1
                car = cars.get(str(numbers[0]))
                assert car is not None, "pilot car source missing"
                summary, found = measure(car, selected[selected.Driver.eq(driver)], origin)
                rows.append(dict(season=season, round=round_, driver=driver, **summary))
                events.append(found.assign(season=season, round=round_, driver=driver))
            report["available"] = True
        reports.append(report)
        print("SOURCE", json.dumps(report), flush=True)
    support = pd.DataFrame(rows, columns=["season", "round", "driver", "native_samples", "eligible", "supported_gears", "marked", "episodes"])
    episodes = pd.concat(events, ignore_index=True) if events else pd.DataFrame(columns=EVENT_COLS+["season", "round", "driver"])
    files = {}
    for name, frame in [("driver_support", support), ("episodes", episodes)]:
        path = ROOT/f"results/e258_{name}.csv"
        frame.to_csv(path, index=False)
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(reports=reports, gate=gate(support, episodes, reports), episodes=len(episodes),
                  episode_drivers=int(episodes.driver.nunique()), winner_fit=False, files=files)
    (ROOT/"results/e258_source_summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print("RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
