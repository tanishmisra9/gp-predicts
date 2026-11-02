"""Energy-deployment signature per driver from the fastest qualifying lap -> data/parts/energy_<first>_<last>.csv.
acc_mid: mean full-throttle acceleration 200-260 km/h | acc_hi: 280-330 km/h | taper = acc_mid - acc_hi (how fast deployment fades with speed)
clip_share: fraction of full-throttle time above 270 km/h with acceleration < 1 m/s^2 (depleted battery / drag-limited) | sp_end: speed at end of the longest straight vs its peak."""
import os, sys, warnings
import fastf1, numpy as np, pandas as pd
from f1.data import DATA
warnings.filterwarnings("ignore"); fastf1.Cache.enable_cache(str(DATA / "fastf1_cache"))

def lapstats(car):
    t = car.Time.dt.total_seconds().values; v = car.Speed.values / 3.6; sp = car.Speed.values; a = np.gradient(v, t); th = car.Throttle.values
    ok = (th >= 98) & np.isfinite(a) & (np.abs(a) < 15) & (car.DRS.values < 10)
    mid = ok & (sp >= 200) & (sp <= 260); hi = ok & (sp >= 280) & (sp <= 330); fast = ok & (sp > 270)
    r = dict(acc_mid=a[mid].mean() if mid.sum() > 15 else np.nan, acc_hi=a[hi].mean() if hi.sum() > 15 else np.nan, clip_share=(a[fast] < 1.0).mean() if fast.sum() > 20 else np.nan)
    r["taper"] = r["acc_mid"] - r["acc_hi"]; return r

def weekend(s):
    rows = []
    for drv in s.laps.Driver.unique():
        try:
            lp = s.laps[s.laps.Driver == drv].pick_fastest()
            if lp is not None and pd.notna(lp.LapTime): rows.append(dict(code=drv, **lapstats(lp.get_car_data())))
        except Exception: pass
    return pd.DataFrame(rows)

def build(first, last):
    f = DATA / f"parts/energy_{first}_{last}.csv"; f.parent.mkdir(exist_ok=True); rows = pd.read_csv(f) if f.exists() else pd.DataFrame()
    done = set(map(tuple, rows[["season", "round"]].drop_duplicates().values)) if len(rows) else set()
    for y in range(first, last + 1):
        for _, ev in fastf1.get_event_schedule(y, include_testing=False).iterrows():
            rnd = int(ev.RoundNumber)
            if (y, rnd) in done or pd.Timestamp(ev.EventDate) >= pd.Timestamp.today().normalize() + pd.Timedelta(days=1): continue
            try:
                s = fastf1.get_session(y, rnd, "Q"); s.load(laps=True, telemetry=True, weather=False, messages=False)
                rows = pd.concat([rows, weekend(s).assign(season=y, round=rnd)]); rows.to_csv(f, index=False); print(y, rnd, "ok", flush=True)
            except Exception as e: print("FAIL", y, rnd, type(e).__name__, str(e)[:80], flush=True)

if __name__ == "__main__":
    if not (os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD")): sys.exit("Set F1_USERNAME / F1_PASSWORD")
    build(int(sys.argv[1]), int(sys.argv[2]))
