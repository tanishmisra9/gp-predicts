"""Start skill + dominance per driver-race -> data/parts/start_<first>_<last>.csv: position after lap 1, laps led (from cached FastF1 race laps)."""
import os, sys, warnings
import fastf1
import pandas as pd
from f1.data import DATA
warnings.filterwarnings("ignore"); fastf1.Cache.enable_cache(str(DATA / "fastf1_cache"))

def build(first, last):
    f = DATA / f"parts/start_{first}_{last}.csv"; f.parent.mkdir(exist_ok=True)
    rows = pd.read_csv(f) if f.exists() else pd.DataFrame()
    done = set(map(tuple, rows[["season", "round"]].drop_duplicates().values)) if len(rows) else set()
    for y in range(first, last + 1):
        for _, ev in fastf1.get_event_schedule(y, include_testing=False).iterrows():
            rnd = int(ev.RoundNumber)
            if (y, rnd) in done or pd.Timestamp(ev.EventDate) >= pd.Timestamp.today().normalize() + pd.Timedelta(days=1): continue
            try:
                s = fastf1.get_session(y, rnd, "R"); s.load(laps=True, telemetry=False, weather=False, messages=False); l = s.laps
                g = l.groupby("Driver"); n = l.LapNumber.max()
                o = pd.DataFrame({"pos_lap1": l[l.LapNumber == 1].set_index("Driver").Position, "laps_led": g.Position.apply(lambda p: (p == 1).sum())})
                o["led_share"] = o.laps_led / n
                rows = pd.concat([rows, o.reset_index(names="code").assign(season=y, round=rnd)]); rows.to_csv(f, index=False); print(y, rnd, "ok", flush=True)
            except Exception as e: print("FAIL", y, rnd, type(e).__name__, str(e)[:80], flush=True)

if __name__ == "__main__":
    if not (os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD")): sys.exit("Set F1_USERNAME / F1_PASSWORD")
    build(int(sys.argv[1]), int(sys.argv[2]))
