"""Alternative race-pace definitions from cached FastF1 race laps -> data/parts/pacev_<first>_<last>.csv (gap % to best driver for each definition).
med: median clean lap (current) | p25: 25th-pct clean lap (less traffic) | fcorr: median after fuel correction (0.035 s/lap) | stint1: laps 2-12 median | best_stint: fastest 8-lap rolling mean
| tcorr: median after removing a per-compound tyre-age slope."""
import os, sys, warnings
import fastf1, numpy as np, pandas as pd
from f1.data import DATA
warnings.filterwarnings("ignore"); fastf1.Cache.enable_cache(str(DATA / "fastf1_cache"))

def variants(laps):
    l = laps[(laps.LapNumber > 1) & laps.LapTime.notna() & laps.PitInTime.isna() & laps.PitOutTime.isna() & laps.TrackStatus.astype(str).eq("1") & laps.IsAccurate.fillna(False)].copy()
    l["t"] = l.LapTime.dt.total_seconds(); l = l[l.t < 1.07 * l.groupby("Driver").t.transform("median")]; nl = laps.LapNumber.max(); rows = {}
    for drv, g in l.groupby("Driver"):
        if len(g) < 8: continue
        g = g.sort_values("LapNumber"); t = g.t.values; ln = g.LapNumber.values
        r = dict(med=np.median(t), p25=np.percentile(t, 25), fcorr=np.median(t + 0.035 * ln))   # fuel burn makes later laps faster by ~0.035 s/lap: add it back to compare at equal fuel
        s1 = g[g.LapNumber <= 12]; r["stint1"] = np.median(s1.t) if len(s1) >= 6 else np.nan
        r["best_stint"] = pd.Series(t).rolling(8).mean().min() if len(t) >= 8 else np.nan
        if g.TyreLife.notna().all() and len(g) >= 12:
            t2 = t.copy()
            for _, cg in g.groupby("Compound"):
                if len(cg) >= 6:
                    b = np.polyfit(cg.TyreLife.values, cg.t.values, 1)[0]; t2[g.index.get_indexer(cg.index)] = cg.t.values - b * (cg.TyreLife.values - cg.TyreLife.values.mean())
            r["tcorr"] = np.median(t2)
        else: r["tcorr"] = np.nan
        rows[drv] = r
    o = pd.DataFrame(rows).T
    for c in list(o): o[c + "_gap"] = (o[c] / o[c].min() - 1) * 100
    return o[[c for c in o if c.endswith("_gap")]]

def build(first, last):
    f = DATA / f"parts/pacev_{first}_{last}.csv"; f.parent.mkdir(exist_ok=True); rows = pd.read_csv(f) if f.exists() else pd.DataFrame()
    done = set(map(tuple, rows[["season", "round"]].drop_duplicates().values)) if len(rows) else set()
    for y in range(first, last + 1):
        for _, ev in fastf1.get_event_schedule(y, include_testing=False).iterrows():
            rnd = int(ev.RoundNumber)
            if (y, rnd) in done or pd.Timestamp(ev.EventDate) >= pd.Timestamp.today().normalize() + pd.Timedelta(days=1): continue
            try:
                s = fastf1.get_session(y, rnd, "R"); s.load(laps=True, telemetry=False, weather=False, messages=False)
                rows = pd.concat([rows, variants(s.laps).reset_index(names="code").assign(season=y, round=rnd)]); rows.to_csv(f, index=False); print(y, rnd, "ok", flush=True)
            except Exception as e: print("FAIL", y, rnd, type(e).__name__, str(e)[:80], flush=True)

if __name__ == "__main__":
    if not (os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD")): sys.exit("Set F1_USERNAME / F1_PASSWORD")
    build(int(sys.argv[1]), int(sys.argv[2]))
