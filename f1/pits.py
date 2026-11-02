"""Pit execution per driver-race -> data/parts/pits_<first>_<last>.csv: median pit-lane time (PitOutTime-PitInTime, red-flag stops dropped), stops, first-stop lap."""
import os
import sys
import warnings

import fastf1
import numpy as np
import pandas as pd

from f1.data import DATA

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache(str(DATA / "fastf1_cache"))


def pits(laps):
    out = []
    for drv, d in laps.groupby("Driver"):
        d = d.sort_values("LapNumber")
        pin = d[d.PitInTime.notna()][["LapNumber", "PitInTime"]]; pout = d[d.PitOutTime.notna()][["LapNumber", "PitOutTime"]]
        t = []
        for _, r in pin.iterrows():
            nxt = pout[pout.LapNumber == r.LapNumber + 1]
            if len(nxt):
                dt = (nxt.PitOutTime.iloc[0] - r.PitInTime).total_seconds()
                if 10 < dt < 60:  # drop red-flag / drive-through / stop-go outliers
                    t.append(dt)
        out.append(dict(code=drv, pit_time=np.median(t) if t else np.nan, n_pit=len(pin), first_stop=pin.LapNumber.min() if len(pin) else np.nan))
    o = pd.DataFrame(out)
    o["pit_gap"] = o.pit_time - o.pit_time.median()  # seconds slower than the field's median stop this race
    return o


def build(first, last):
    f = DATA / f"parts/pits_{first}_{last}.csv"; f.parent.mkdir(exist_ok=True)
    rows = pd.read_csv(f) if f.exists() else pd.DataFrame()
    done = set(map(tuple, rows[["season", "round"]].drop_duplicates().values)) if len(rows) else set()
    for y in range(first, last + 1):
        for _, ev in fastf1.get_event_schedule(y, include_testing=False).iterrows():
            rnd = int(ev.RoundNumber)
            if (y, rnd) in done or pd.Timestamp(ev.EventDate) >= pd.Timestamp.today().normalize() + pd.Timedelta(days=1):
                continue
            try:
                s = fastf1.get_session(y, rnd, "R"); s.load(laps=True, telemetry=False, weather=False, messages=False)
                rows = pd.concat([rows, pits(s.laps).assign(season=y, round=rnd)]); rows.to_csv(f, index=False); print(y, rnd, "ok", flush=True)
            except Exception as e:
                print("FAIL", y, rnd, type(e).__name__, str(e)[:100], flush=True)


if __name__ == "__main__":
    if not (os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD")):
        sys.exit("Set F1_USERNAME / F1_PASSWORD")
    build(int(sys.argv[1]), int(sys.argv[2]))
