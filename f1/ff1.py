"""FastF1 race-pace ingestion -> data/racepace_ff1.csv  (season, round, code, pace_gap).
pace_gap = driver's median clean green-flag lap, % slower than the race's best driver. Needs F1-account credentials in the
environment (F1_USERNAME / F1_PASSWORD; FastF1>=3.9). NOT used for results: FastF1's grid/points/status come from Jolpica internally
(NaN when rate-limited) and its timing-based positions ignore post-race penalties, so official results stay on Jolpica (f1/data.py)."""
import os
import sys
import warnings

import fastf1
import pandas as pd

from f1.data import DATA

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache(str(DATA / "fastf1_cache"))


def race_pace(laps):
    l = laps[(laps.LapNumber > 1) & laps.LapTime.notna() & laps.PitInTime.isna() & laps.PitOutTime.isna()
             & laps.TrackStatus.astype(str).eq("1") & laps.IsAccurate.fillna(False)].copy()
    l["t"] = l.LapTime.dt.total_seconds()
    l = l[l.t < 1.07 * l.groupby("Driver").t.transform("median")]
    g = l.groupby("Driver").t
    p = g.median().where(g.size() >= 8)
    return ((p / p.min() - 1) * 100).rename("pace_gap")


def build(first=2018, last=2026):
    tag = f"{first}_{last}"  # one output per season range so ranges can be ingested in parallel
    fp = DATA / f"parts/racepace_ff1_{tag}.csv"; fp.parent.mkdir(exist_ok=True)
    pace = [pd.read_csv(fp)] if fp.exists() else []
    done = {(r.season, r["round"]) for _, r in pace[0].iterrows()} if pace else set()
    for y in range(first, last + 1):
        for _, ev in fastf1.get_event_schedule(y, include_testing=False).iterrows():
            rnd = int(ev.RoundNumber)
            if (y, rnd) in done or pd.Timestamp(ev.EventDate) >= pd.Timestamp.today().normalize() + pd.Timedelta(days=1):
                continue
            try:
                s = fastf1.get_session(y, rnd, "R"); s.load(laps=True, telemetry=False, weather=False, messages=False)
                pace.append(race_pace(s.laps).reset_index().rename(columns={"Driver": "code"}).assign(season=y, round=rnd))
            except Exception as e:
                print("FAIL", y, rnd, type(e).__name__, str(e)[:100], flush=True); continue
            pd.concat(pace).to_csv(fp, index=False); print(y, rnd, ev.EventName, "ok", flush=True)


def merge():
    d = pd.concat([pd.read_csv(f) for f in sorted((DATA / "parts").glob("racepace_ff1_*.csv"))])
    d.drop_duplicates(["season", "round", "code"]).sort_values(["season", "round"]).to_csv(DATA / "racepace_ff1.csv", index=False)


if __name__ == "__main__":
    if sys.argv[1:] == ["merge"]:
        merge(); sys.exit()
    if not (os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD")):
        sys.exit("Set F1_USERNAME / F1_PASSWORD (FastF1>=3.9 needs an F1 account for live-timing data).")
    a = sys.argv[1:3]
    build(*(int(x) for x in a)) if a else build()
