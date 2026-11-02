"""Race-strategy character per race -> data/parts/strat_<first>_<last>.csv. Aggregated by circuit (PAST visits only) in features to describe a track as a
'typical 1-stop' or '2-stop' race, its tyre usage, how often it rains, and how often safety cars appear."""
import os
import sys
import warnings

import fastf1
import numpy as np
import pandas as pd

from f1.data import DATA

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache(str(DATA / "fastf1_cache"))


def race_strategy(laps):
    nl = laps.LapNumber.max()
    d = laps.groupby("Driver").agg(stops=("Stint", lambda s: s.max() - 1), nlap=("LapNumber", "max"))
    run = d[d.nlap >= 0.9 * nl]  # drivers who ran (nearly) the full distance: their stop count reflects the strategy
    comp = laps.Compound.value_counts(normalize=True)
    ts = laps.TrackStatus.astype(str)
    sc = ts.str.contains("4").groupby(laps.LapNumber).any().mean(); vsc = ts.str.contains("[67]").groupby(laps.LapNumber).any().mean()
    stints = laps.groupby(["Driver", "Stint"]).size()
    return dict(stops_med=run.stops.median(), stops_2plus=(run.stops >= 2).mean(), stops_mean=run.stops.mean(),
                stint_len=stints.median(), soft=comp.get("SOFT", 0), medium=comp.get("MEDIUM", 0), hard=comp.get("HARD", 0),
                wet=comp.get("INTERMEDIATE", 0) + comp.get("WET", 0), sc_frac=sc, vsc_frac=vsc, race_laps=nl)


def build(first, last):
    f = DATA / f"parts/strat_{first}_{last}.csv"; f.parent.mkdir(exist_ok=True)
    rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    done = {(r["season"], r["round"]) for r in rows}
    for y in range(first, last + 1):
        for _, ev in fastf1.get_event_schedule(y, include_testing=False).iterrows():
            rnd = int(ev.RoundNumber)
            if (y, rnd) in done or pd.Timestamp(ev.EventDate) >= pd.Timestamp.today().normalize() + pd.Timedelta(days=1):
                continue
            try:
                s = fastf1.get_session(y, rnd, "R"); s.load(laps=True, telemetry=False, weather=False, messages=False)
                rows.append(dict(season=y, round=rnd, **race_strategy(s.laps)))
                pd.DataFrame(rows).to_csv(f, index=False); print(y, rnd, ev.EventName, "ok", flush=True)
            except Exception as e:
                print("FAIL", y, rnd, type(e).__name__, str(e)[:100], flush=True)


if __name__ == "__main__":
    if not (os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD")):
        sys.exit("Set F1_USERNAME / F1_PASSWORD")
    build(int(sys.argv[1]), int(sys.argv[2]))
