"""Pre-season testing pace per driver -> data/parts/testing_<first>_<last>.csv: best-lap gap and long-run gap vs the best driver in each testing session, averaged over sessions."""
import os, sys, warnings
import fastf1, numpy as np, pandas as pd
from f1.data import DATA
from f1.sessions import _longrun
warnings.filterwarnings("ignore"); fastf1.Cache.enable_cache(str(DATA / "fastf1_cache"))

def build(first, last):
    f = DATA / f"parts/testing_{first}_{last}.csv"; f.parent.mkdir(exist_ok=True); rows = []
    for y in range(first, last + 1):
        sched = fastf1.get_event_schedule(y, include_testing=True); tests = sched[sched.RoundNumber == 0]
        for ti, (_, ev) in enumerate(tests.iterrows(), start=1):
            for i in range(1, 4):
                try:
                    s = fastf1.get_session(y, ti, f"Practice {i}") if False else fastf1.get_testing_event(y, ti).get_session(i); s.load(laps=True, telemetry=False, weather=False, messages=False)
                    l = s.laps; l = l[l.LapTime.notna()]
                    b = l.groupby("Driver").LapTime.min().dt.total_seconds(); lr = l.groupby("Driver").apply(_longrun, include_groups=False)
                    o = pd.DataFrame({"best": b, "lr": lr}); 
                    for c in ("best", "lr"): o[c + "_gap"] = (o[c] / o[c].min() - 1) * 100
                    rows.append(o[["best_gap", "lr_gap"]].reset_index(names="code").assign(season=y, test=ti, day=i)); print(y, ti, i, "ok", flush=True)
                except Exception as e: print("FAIL", y, ti, i, type(e).__name__, str(e)[:80], flush=True)
    if rows: pd.concat(rows).to_csv(f, index=False)

if __name__ == "__main__":
    if not (os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD")): sys.exit("Set F1_USERNAME / F1_PASSWORD")
    build(int(sys.argv[1]), int(sys.argv[2]))
