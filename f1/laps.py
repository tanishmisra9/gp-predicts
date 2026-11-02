"""OpenF1 lap summaries -> data/laps.csv (one row per season/round/session/driver).
Why OpenF1 and not fastf1: fastf1 3.9 needs F1-account credentials for live-timing data and 3.3.8 can't parse
2026 sessions; OpenF1 serves the same timing feed unauthenticated (2023+).
Everything is expressed as a % gap to the best driver in that same session, so tracks/years are comparable."""
import time

import numpy as np
import pandas as pd
import requests_cache

from f1.data import DATA, load_results

API = "https://api.openf1.org/v1"
_s = requests_cache.CachedSession(str(DATA / "openf1_cache"), expire_after=None)
SES = {"Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3", "Sprint Qualifying": "SQ",
       "Sprint": "SPR", "Qualifying": "Q", "Race": "R"}


def get(path, **params):
    for attempt in range(8):
        r = _s.get(f"{API}/{path}", params=params, timeout=60)
        if not getattr(r, "from_cache", False):
            time.sleep(0.6)  # observed: bursts of 4 pass; 429 handler below backs off if the limit bites
        if r.status_code == 429:
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code == 404:  # OpenF1 returns 404 for "no data"
            return []
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"OpenF1 rate-limited: {path} {params}")


def _mark_clean(l):
    """Keep green-flag, in-traffic-agnostic racing laps: drop lap 1, in/out laps, neutralised (SC/VSC) laps, outliers."""
    l = l.sort_values(["driver_number", "lap_number"]).copy()
    l["next_out"] = l.groupby("driver_number")["is_pit_out_lap"].shift(-1).fillna(False).astype(bool)
    l = l[(l.lap_number > 1) & ~l.is_pit_out_lap.astype(bool) & ~l.next_out & l.lap_duration.notna()]
    fm = l.groupby("lap_number")["lap_duration"].median()  # field median per lap number
    neutral = fm > 1.07 * fm.median()
    l = l[~l.lap_number.map(neutral).fillna(False)]
    med = l.groupby("driver_number")["lap_duration"].transform("median")
    return l[l.lap_duration < 1.07 * med]


def summarize(laps, kind):
    """kind: 'run' = practice/quali (best + top-5 mean), 'race' = clean-lap median pace."""
    laps = pd.DataFrame(laps)
    if laps.empty or "lap_duration" not in laps:
        return pd.DataFrame()
    laps = laps[laps.lap_duration.notna()]
    if kind == "race":
        c = _mark_clean(laps)
        g = c.groupby("driver_number").lap_duration
        out = pd.DataFrame({"pace": g.median(), "n": g.size()})
        out.loc[out.n < 8, "pace"] = np.nan
        out["pace_gap"] = (out.pace / out.pace.min() - 1) * 100
        return out[["pace_gap", "n"]]
    l = laps[~laps.is_pit_out_lap.astype(bool)]
    g = l.groupby("driver_number").lap_duration
    out = pd.DataFrame({"best": g.min(), "top5": g.apply(lambda x: x.nsmallest(5).mean()), "n": g.size()})
    out["lr"] = l.groupby("driver_number")[["lap_number", "lap_duration"]].apply(_longrun)
    for c in ("best", "top5", "lr"):
        out[c + "_gap"] = (out[c] / out[c].min() - 1) * 100
    return out[["best_gap", "top5_gap", "lr_gap", "n"]]


def _longrun(d):
    """Median lap of the driver's longest run (>=5) of consecutive laps within 110% of his best = race-sim pace."""
    d = d.sort_values("lap_number")
    ok = d[d.lap_duration < 1.10 * d.lap_duration.min()]
    seg = (ok.lap_number.diff() != 1).cumsum()
    runs = [x for _, x in ok.groupby(seg) if len(x) >= 5]
    return max(runs, key=len).lap_duration.median() if runs else np.nan


def build(first=2023, last=2026):
    res = load_results()
    races = res.drop_duplicates(["season", "round"])[["season", "round", "date"]]
    rows = []
    for year in range(first, last + 1):
        ss = pd.DataFrame(get("sessions", year=year))
        ss["day"] = pd.to_datetime(ss.date_start).dt.tz_localize(None).dt.normalize()
        rr = ss[ss.session_name == "Race"]
        for _, m in rr.iterrows():
            cand = races[(races.season == year) & ((pd.to_datetime(races.date) - m.day).abs() <= pd.Timedelta(days=1))]
            if cand.empty:
                continue  # future / cancelled round without Jolpica results
            rnd = int(cand.iloc[0]["round"])
            code = {d["driver_number"]: d["name_acronym"] for d in get("drivers", meeting_key=int(m.meeting_key))}
            for _, s in ss[ss.meeting_key == m.meeting_key].iterrows():
                if s.session_name not in SES:
                    continue
                laps = get("laps", session_key=int(s.session_key))
                out = summarize(laps, "race" if s.session_name in ("Race", "Sprint") else "run")
                for dn, r in out.iterrows():
                    rows.append(dict(season=year, round=rnd, session=SES[s.session_name], code=code.get(dn), **r.to_dict()))
            print(year, rnd, "ok", flush=True)
            pd.DataFrame(rows).to_csv(DATA / "laps.csv", index=False)  # incremental: safe to interrupt/resume
    return pd.DataFrame(rows)


if __name__ == "__main__":
    build()
