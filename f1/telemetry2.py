"""Telemetry v2: corner/engine profile averaged over each driver's 3 FASTEST qualifying laps (v1 used one lap) -> data/parts/tel2_<first>_<last>.csv.
Same metric definitions as f1/telemetry.py; averaging should cut the lap-to-lap noise that the reliability audit (E20) found to dominate v1."""
import os, sys, warnings
import fastf1, numpy as np, pandas as pd
from f1.data import DATA
from f1.telemetry import _grid, _corners, N
warnings.filterwarnings("ignore"); fastf1.Cache.enable_cache(str(DATA / "fastf1_cache"))


def weekend2(s, nlaps=3):
    laps = s.laps; ent = {}                       # (driver, lap-rank) -> (laptime, grid)
    for drv in laps.Driver.unique():
        d = laps[(laps.Driver == drv) & laps.LapTime.notna()].nsmallest(nlaps, "LapTime")
        for i, (_, lp) in enumerate(d.iterrows()):
            try: ent[(drv, i)] = (lp.LapTime.total_seconds(), _grid(lp.get_car_data()))
            except Exception: pass
    ref = min(ent, key=lambda k: ent[k][0]); gr = ent[ref][1]; mins, b = _corners(gr["v"]); step = gr["L"] / N
    cls = np.where(gr["v"][mins] < 130, "slow", np.where(gr["v"][mins] < 220, "med", "fast")); per = {}
    for key, (_, g) in ent.items():
        v, th = g["v"], g["th"]; rows = []
        for i, m in enumerate(mins):
            lo, hi = b[i], b[i + 1]; apex = v[max(m - 6, 0):m + 7].min(); entry = v[lo:m + 1].max(); dist = max((m - lo) * step, 1.0)
            decel = (entry ** 2 - apex ** 2) / 3.6 ** 2 / (2 * dist); after = th[m:min(m + 60, N)]; pick = np.argmax(after >= 95) * step if (after >= 95).any() else 300.0
            rows.append((cls[i], apex, decel, min(pick, 300.0), v[min(m + int(150 / step), N - 1)] - apex, v[m:hi + 1].max()))
        per[key] = pd.DataFrame(rows, columns=["cls", "apex", "decel", "pick", "gain", "exitv"])
    keys = list(per); out = []
    vmax = max(g["v"].max() for _, g in ent.values()); accs = np.array([ent[k][1]["acc"] for k in keys], float)
    for ix, key in enumerate(keys):
        r = dict(code=key[0], acc_hs=ent[key][1]["acc"], vmax_gap=(1 - ent[key][1]["v"].max() / vmax) * 100)
        for k in ("slow", "med", "fast"):
            sel = [j for j in range(len(mins)) if cls[j] == k]
            if not sel: continue
            A, D, P_, G, Xv = (np.array([per[q][c].values[sel] for q in keys]) for c in ("apex", "decel", "pick", "gain", "exitv"))
            r[f"apex_{k}"] = ((A.max(0) - A[ix]) / A.max(0) * 100).mean(); r[f"brake_{k}"] = (D.max(0) - D[ix]).mean(); r[f"pick_{k}"] = (P_[ix] - P_.min(0)).mean()
            r[f"gain_{k}"] = (G.max(0) - G[ix]).mean(); r[f"exitv_{k}"] = ((Xv.max(0) - Xv[ix]) / Xv.max(0) * 100).mean()
        r["acc_gap"] = np.nanmax(accs) - r["acc_hs"] if np.isfinite(r["acc_hs"]) else np.nan; out.append(r)
    return pd.DataFrame(out).groupby("code").mean(numeric_only=True).reset_index()   # average the driver's 3 laps


def build(first, last):
    f = DATA / f"parts/tel2_{first}_{last}.csv"; f.parent.mkdir(exist_ok=True); rows = pd.read_csv(f) if f.exists() else pd.DataFrame()
    done = set(map(tuple, rows[["season", "round"]].drop_duplicates().values)) if len(rows) else set()
    for y in range(first, last + 1):
        for _, ev in fastf1.get_event_schedule(y, include_testing=False).iterrows():
            rnd = int(ev.RoundNumber)
            if (y, rnd) in done or pd.Timestamp(ev.EventDate) >= pd.Timestamp.today().normalize() + pd.Timedelta(days=1): continue
            try:
                s = fastf1.get_session(y, rnd, "Q"); s.load(laps=True, telemetry=True, weather=False, messages=False)
                rows = pd.concat([rows, weekend2(s).assign(season=y, round=rnd)]); rows.to_csv(f, index=False); print(y, rnd, "ok", flush=True)
            except Exception as e: print("FAIL", y, rnd, type(e).__name__, str(e)[:90], flush=True)


if __name__ == "__main__":
    if not (os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD")): sys.exit("Set F1_USERNAME / F1_PASSWORD")
    build(int(sys.argv[1]), int(sys.argv[2]))
