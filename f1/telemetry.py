"""Corner-profile + engine telemetry from each driver's fastest qualifying lap -> data/parts/tel_<first>_<last>.csv (+ circuit traits).
Per weekend a reference lap (session-fastest) defines corners (speed minima). Every lap is resampled onto the same track-position grid so
corners align. Per corner and driver: apex speed, braking decel into the corner, distance to full throttle after apex, speed gained over the
next 150 m, and the top speed on the following straight. Gaps vs the best car are averaged by corner class (slow<130, medium<220, fast).
Engine proxy: full-throttle acceleration (DRS closed) at 200-280 km/h, plus straight-end speeds."""
import os
import sys
import warnings

import fastf1
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from f1.data import DATA

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache(str(DATA / "fastf1_cache"))
N = 1000  # resample points per lap


def _grid(car):
    """resample speed/throttle/brake onto N equally spaced track-position points; also return lap length (m) and full-throttle accel."""
    t = car.Time.dt.total_seconds().values; v = car.Speed.values / 3.6
    d = np.concatenate([[0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))]); L = d[-1]
    u = np.linspace(0, L, N)
    g = {k: np.interp(u, d, car[c].astype(float).values) for k, c in (("v", "Speed"), ("th", "Throttle"), ("br", "Brake"))}
    # high-speed full-throttle acceleration, DRS closed (power vs drag): dv/dt over samples with throttle>=98, 200-280 km/h
    a = np.gradient(v, t); sp = car.Speed.values
    m = (car.Throttle.values >= 98) & (sp >= 200) & (sp <= 280) & (car.DRS.values < 10) & np.isfinite(a) & (np.abs(a) < 15)
    g["acc"] = a[m].mean() if m.sum() >= 15 else np.nan
    g["L"] = L
    return g


def _corners(vref):
    vs = np.convolve(vref, np.ones(5) / 5, mode="same")
    mins, _ = find_peaks(-vs, prominence=25, distance=12)  # v in km/h
    b = [0] + [int(m0 + np.argmax(vs[m0:m1])) for m0, m1 in zip(mins[:-1], mins[1:])] + [int(mins[-1] + np.argmax(vs[mins[-1]:]))]
    return mins, b


def weekend(s):
    laps = s.laps; best = {}
    for drv in laps.Driver.unique():
        try:
            lp = laps[laps.Driver == drv].pick_fastest()
            if lp is not None and pd.notna(lp.LapTime):
                best[drv] = (lp.LapTime.total_seconds(), _grid(lp.get_car_data()))
        except Exception:
            pass
    ref = min(best, key=lambda k: best[k][0]); gr = best[ref][1]
    mins, b = _corners(gr["v"]); step = gr["L"] / N
    cls = np.where(gr["v"][mins] < 130, "slow", np.where(gr["v"][mins] < 220, "med", "fast"))
    per = {}
    for drv, (_, g) in best.items():
        v, th = g["v"], g["th"]; rows = []
        for i, m in enumerate(mins):
            lo, hi = b[i], b[i + 1]
            apex = v[max(m - 6, 0):m + 7].min(); entry = v[lo:m + 1].max()
            dist = max((m - lo) * step, 1.0)
            decel = (entry ** 2 - apex ** 2) / 3.6 ** 2 / (2 * dist)
            after = th[m:min(m + 60, N)]; pick = np.argmax(after >= 95) * step if (after >= 95).any() else 300.0
            gain = v[min(m + int(150 / step), N - 1)] - apex
            rows.append((cls[i], apex, decel, min(pick, 300.0), gain, v[m:hi + 1].max()))
        per[drv] = pd.DataFrame(rows, columns=["cls", "apex", "decel", "pick", "gain", "exitv"])
    out = []
    ncor = len(mins)
    for drv, c in per.items():
        r = dict(code=drv, acc_hs=best[drv][1]["acc"], vmax_gap=(1 - best[drv][1]["v"].max() / max(g["v"].max() for _, g in best.values())) * 100,
                 ft=(best[drv][1]["th"] >= 98).mean(), brk=(best[drv][1]["br"] > 0.5).mean())
        for k in ("slow", "med", "fast"):
            sel = [j for j in range(ncor) if cls[j] == k]
            if not sel:
                continue
            A = np.array([per[d].apex.values[sel] for d in per]); D = np.array([per[d].decel.values[sel] for d in per])
            P = np.array([per[d].pick.values[sel] for d in per]); G = np.array([per[d].gain.values[sel] for d in per]); X = np.array([per[d].exitv.values[sel] for d in per])
            ix = list(per).index(drv)
            r[f"apex_{k}"] = ((A.max(0) - A[ix]) / A.max(0) * 100).mean(); r[f"brake_{k}"] = (D.max(0) - D[ix]).mean()
            r[f"pick_{k}"] = (P[ix] - P.min(0)).mean(); r[f"gain_{k}"] = (G.max(0) - G[ix]).mean(); r[f"exitv_{k}"] = ((X.max(0) - X[ix]) / X.max(0) * 100).mean()
        out.append(r)
    accs = np.array([r["acc_hs"] for r in out], float)
    for r in out:  # engine proxy as gap to the best car (m/s^2)
        r["acc_gap"] = np.nanmax(accs) - r["acc_hs"] if np.isfinite(r["acc_hs"]) and np.isfinite(accs).any() else np.nan
    # static circuit traits from the reference lap
    thr = gr["th"] >= 98; run = np.diff(np.concatenate([[0], thr.astype(int), [0]])); st, en = np.where(run == 1)[0], np.where(run == -1)[0]
    tr = dict(c_len=gr["L"], c_slow=(cls == "slow").sum(), c_med=(cls == "med").sum(), c_fast=(cls == "fast").sum(), c_ft=thr.mean(),
              c_brk=(gr["br"] > 0.5).mean(), c_vmax=gr["v"].max(), c_vmean=gr["v"].mean(),
              c_longest=(en - st).max() * step if len(st) else 0.0, c_lowspeed=(gr["v"] < 130).mean(), c_ncor=ncor)
    return pd.DataFrame(out), tr


def build(first, last):
    f, ft = DATA / f"parts/tel_{first}_{last}.csv", DATA / f"parts/circ_{first}_{last}.csv"; f.parent.mkdir(exist_ok=True)
    rows = pd.read_csv(f) if f.exists() else pd.DataFrame(); circ = pd.read_csv(ft) if ft.exists() else pd.DataFrame()
    done = set(map(tuple, rows[["season", "round"]].drop_duplicates().values)) if len(rows) else set()
    for y in range(first, last + 1):
        for _, ev in fastf1.get_event_schedule(y, include_testing=False).iterrows():
            rnd = int(ev.RoundNumber)
            if (y, rnd) in done or pd.Timestamp(ev.EventDate) >= pd.Timestamp.today().normalize() + pd.Timedelta(days=1):
                continue
            try:
                s = fastf1.get_session(y, rnd, "Q"); s.load(laps=True, telemetry=True, weather=False, messages=False)
                w, tr = weekend(s)
                rows = pd.concat([rows, w.assign(season=y, round=rnd)]); circ = pd.concat([circ, pd.DataFrame([dict(season=y, round=rnd, **tr)])])
                rows.to_csv(f, index=False); circ.to_csv(ft, index=False); print(y, rnd, ev.EventName, "ok", len(w), flush=True)
            except Exception as e:
                print("FAIL", y, rnd, type(e).__name__, str(e)[:100], flush=True)


if __name__ == "__main__":
    if not (os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD")):
        sys.exit("Set F1_USERNAME / F1_PASSWORD")
    build(int(sys.argv[1]), int(sys.argv[2]))
