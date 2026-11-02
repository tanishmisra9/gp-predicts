"""FastF1 per-session lap summaries for EVERY session of a weekend (FP1/2/3, Q, SQ, S, R) -> data/parts/sessions_<first>_<last>.csv.
One row per (season, round, session, driver). All gaps are % vs the best driver in that session (so tracks/years are comparable).
Columns: best_gap, lr_gap (long-run pace), s1/s2/s3_gap (best sector), st/fl/i1/i2_gap (straight-line speed: % below session max),
st_max (km/h), deg (tyre-degradation slope s/lap, race+sprint only), n (laps)."""
import os
import sys
import warnings

import fastf1
import numpy as np
import pandas as pd

from f1.data import DATA

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache(str(DATA / "fastf1_cache"))
LABEL = {"Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3", "Qualifying": "Q", "Sprint Shootout": "SQ",
         "Sprint Qualifying": "SQ", "Sprint": "S", "Race": "R"}


def _longrun(d):
    d = d.sort_values("LapNumber"); t = d.LapTime.dt.total_seconds()
    ok = d[(t < 1.10 * t.min()) & d.PitInTime.isna() & d.PitOutTime.isna()]
    seg = (ok.LapNumber.diff() != 1).cumsum()
    runs = [x for _, x in ok.groupby(seg) if len(x) >= 5]
    return max(runs, key=len).LapTime.dt.total_seconds().median() if runs else np.nan


def _deg(d):
    """mean slope of lap time vs tyre age over clean stints of >=8 laps (s/lap; positive = tyres getting slower)."""
    sl = []
    for _, st in d[d.TrackStatus.astype(str).eq("1") & d.PitInTime.isna() & d.PitOutTime.isna() & (d.LapNumber > 1)].groupby("Stint"):
        if len(st) >= 8 and st.TyreLife.notna().all():
            t = st.LapTime.dt.total_seconds().values
            m = t < 1.05 * np.median(t)
            if m.sum() >= 8:
                sl.append(np.polyfit(st.TyreLife.values[m], t[m], 1)[0])
    return np.mean(sl) if sl else np.nan


def summarize(laps, kind):
    l = laps[laps.LapTime.notna()].copy()
    if "Deleted" in l:
        l = l[~l.Deleted.fillna(False).astype(bool)]
    out = {}
    for drv, d in l.groupby("Driver"):
        r = dict(n=len(d))
        if kind == "R":
            c = d[(d.LapNumber > 1) & d.PitInTime.isna() & d.PitOutTime.isna() & d.TrackStatus.astype(str).eq("1")]
            t = c.LapTime.dt.total_seconds(); t = t[t < 1.07 * t.median()] if len(t) else t
            r["pace"] = t.median() if len(t) >= 8 else np.nan
            r["deg"] = _deg(d)
        else:
            r["best"] = d.LapTime.dt.total_seconds().min(); r["lr"] = _longrun(d)
        for k, col in (("s1", "Sector1Time"), ("s2", "Sector2Time"), ("s3", "Sector3Time")):
            r[k] = d[col].dt.total_seconds().min() if col in d else np.nan
        for k, col in (("st", "SpeedST"), ("fl", "SpeedFL"), ("i1", "SpeedI1"), ("i2", "SpeedI2")):
            r[k] = d[col].max() if col in d else np.nan
        out[drv] = r
    o = pd.DataFrame(out).T
    for k in ("pace", "best", "lr", "s1", "s2", "s3"):
        if k in o:
            o[k + "_gap"] = (o[k] / o[k].min() - 1) * 100
    for k in ("st", "fl", "i1", "i2"):
        o[k + "_gap"] = (1 - o[k] / o[k].max()) * 100
    o["st_max"] = o["st"]
    o = o.rename(columns={"pace_gap": "best_gap"}) if kind == "R" else o
    cols = [c for c in ("best_gap", "lr_gap", "s1_gap", "s2_gap", "s3_gap", "st_gap", "fl_gap", "i1_gap", "i2_gap", "st_max", "deg", "n") if c in o]
    return o[cols]


def build(first, last):
    f = DATA / f"parts/sessions_{first}_{last}.csv"; f.parent.mkdir(exist_ok=True)
    rows = pd.read_csv(f) if f.exists() else pd.DataFrame()
    done = set(map(tuple, rows[["season", "round"]].drop_duplicates().values)) if len(rows) else set()
    today = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    for y in range(first, last + 1):
        for _, ev in fastf1.get_event_schedule(y, include_testing=False).iterrows():
            rnd = int(ev.RoundNumber)
            if (y, rnd) in done or pd.Timestamp(ev.EventDate) >= today:
                continue
            new = []
            for i in range(1, 6):
                name = ev.get(f"Session{i}")
                if name not in LABEL:
                    continue
                try:
                    s = fastf1.get_session(y, rnd, name); s.load(laps=True, telemetry=False, weather=False, messages=False)
                    new.append(summarize(s.laps, "R" if LABEL[name] in ("R", "S") else "run").reset_index(names="code").assign(
                        season=y, round=rnd, session=LABEL[name]))
                except Exception as e:
                    print("FAIL", y, rnd, name, type(e).__name__, str(e)[:80], flush=True)
            if new:
                rows = pd.concat([rows] + new); rows.to_csv(f, index=False)
            print(y, rnd, ev.EventName, len(new), "sessions ok", flush=True)


if __name__ == "__main__":
    if not (os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD")):
        sys.exit("Set F1_USERNAME / F1_PASSWORD")
    build(int(sys.argv[1]), int(sys.argv[2]))
