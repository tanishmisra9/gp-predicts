"""FastF1 ingestion -> data/results_ff1.csv (same schema as the old Jolpica table) + data/racepace_ff1.csv.
Needs F1 account credentials in the environment: F1_USERNAME / F1_PASSWORD (FastF1 >= 3.9 requirement)."""
import os
import sys
import warnings

import fastf1
import numpy as np
import pandas as pd

from f1.data import DATA

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache(str(DATA / "fastf1_cache"))
# team lineage so renamed teams keep their history (FastF1/Ergast constructor ids change on rebrand)
LINEAGE = {"toro_rosso": "rb", "alphatauri": "rb", "racing_point": "aston_martin", "force_india": "aston_martin",
           "renault": "alpine", "alfa": "sauber", "kick_sauber": "sauber", "audi": "sauber", "rb": "rb"}


def _sec(td):
    return td.total_seconds() if pd.notna(td) else np.nan


def _load(year, rnd, name, laps=False):
    s = fastf1.get_session(year, rnd, name)
    s.load(laps=laps, telemetry=False, weather=False, messages=False)
    return s


def race_pace(laps):
    """Clean green-flag lap median per driver, % gap to the best driver (same definition as the OpenF1 version)."""
    l = laps[(laps.LapNumber > 1) & laps.LapTime.notna() & laps.PitInTime.isna() & laps.PitOutTime.isna()
             & laps.TrackStatus.astype(str).eq("1") & laps.IsAccurate.fillna(False)].copy()
    l["t"] = l.LapTime.dt.total_seconds()
    l = l[l.t < 1.07 * l.groupby("Driver").t.transform("median")]
    g = l.groupby("Driver").t
    p = g.median().where(g.size() >= 8)
    return ((p / p.min() - 1) * 100).rename("pace_gap")


def weekend(year, ev):
    rnd = int(ev.RoundNumber)
    race = _load(year, rnd, "R", laps=True)
    res = race.results
    q = _load(year, rnd, "Q").results.set_index("Abbreviation")
    sp = pd.DataFrame()
    if "sprint" in str(ev.EventFormat):
        try:
            sp = _load(year, rnd, "S").results.set_index("Abbreviation")
        except Exception:
            pass
    rows = []
    for _, x in res.iterrows():
        a = x.Abbreviation; qq = q.loc[a] if a in q.index else None; ss = sp.loc[a] if a in sp.index else None
        tid = str(x.TeamId)
        rows.append(dict(
            season=year, round=rnd, date=pd.Timestamp(ev.EventDate).date().isoformat(), race=ev.EventName,
            circuit=str(ev.Location), driver=a, code=a, team=LINEAGE.get(tid, tid),
            grid=int(x.GridPosition) if pd.notna(x.GridPosition) else 0,
            finish=x.Position if str(x.ClassifiedPosition).isdigit() else np.nan, order=int(x.Position),
            status=x.Status, points=float(x.Points), laps=int(x.Laps) if pd.notna(x.Laps) else 0,
            q_pos=qq.Position if qq is not None else np.nan,
            q1=_sec(qq.Q1) if qq is not None else np.nan, q2=_sec(qq.Q2) if qq is not None else np.nan,
            q3=_sec(qq.Q3) if qq is not None else np.nan,
            s_grid=ss.GridPosition if ss is not None else np.nan, s_order=ss.Position if ss is not None else np.nan,
            s_points=ss.Points if ss is not None else np.nan))
    return rows, race_pace(race.laps).reset_index().assign(season=year, round=rnd).rename(columns={"Driver": "code"})


def build(first=2018, last=2026):
    f, fp = DATA / "results_ff1.csv", DATA / "racepace_ff1.csv"
    rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    pace = [pd.read_csv(fp)] if fp.exists() else []
    done = {(r["season"], r["round"]) for r in rows}
    today = pd.Timestamp.today().normalize()
    for y in range(first, last + 1):
        sched = fastf1.get_event_schedule(y, include_testing=False)
        for _, ev in sched.iterrows():
            if (y, int(ev.RoundNumber)) in done or pd.Timestamp(ev.EventDate) >= today + pd.Timedelta(days=1):
                continue
            try:
                r, p = weekend(y, ev)
            except Exception as e:
                print("FAIL", y, ev.RoundNumber, type(e).__name__, str(e)[:100], flush=True); continue
            rows += r; pace.append(p)
            pd.DataFrame(rows).to_csv(f, index=False); pd.concat(pace).to_csv(fp, index=False)  # incremental
            print(y, ev.RoundNumber, ev.EventName, "ok", flush=True)


if __name__ == "__main__":
    if not (os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD")):
        sys.exit("Set F1_USERNAME / F1_PASSWORD (FastF1>=3.9 needs an F1 account for live-timing data).")
    build()
