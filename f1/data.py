"""Data layer: Jolpica (Ergast successor) results/quali/sprint -> one tidy parquet-free CSV per table.
All HTTP is cached on disk (requests-cache), so reruns are offline and deterministic."""
import time
from pathlib import Path

import pandas as pd
import requests_cache

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
BASE = "https://api.jolpi.ca/ergast/f1"
# expire_after=None: cache forever; delete data/http_cache.sqlite (or use refresh) to pull new rounds
_s = requests_cache.CachedSession(str(DATA / "http_cache"), expire_after=None)


def _pages(path):
    """Jolpica caps limit at 100 -> paginate. Yields Race dicts (merged across pages by round)."""
    races, off = {}, 0
    while True:
        r = _s.get(f"{BASE}/{path}.json", params={"limit": 100, "offset": off}, timeout=30)
        if not getattr(r, "from_cache", False):
            time.sleep(0.4)  # polite: Jolpica burst limit
        r.raise_for_status()
        md = r.json()["MRData"]
        for race in md["RaceTable"]["Races"]:
            key = (int(race["season"]), int(race["round"]))
            if key in races:
                for k in ("Results", "QualifyingResults", "SprintResults"):
                    if k in race:
                        races[key][k] = races[key].get(k, []) + race[k]
            else:
                races[key] = race
        off += 100
        if off >= int(md["total"]):
            return list(races.values())


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _qsec(t):
    if not t or ":" not in t:  # missing / "" / non-time placeholders -> no lap set
        return None
    m, s = t.split(":")
    return int(m) * 60 + float(s)


def fetch_season(year):
    res = _pages(f"{year}/results")
    q = {(int(r["season"]), int(r["round"])): r.get("QualifyingResults", []) for r in _pages(f"{year}/qualifying")}
    sp = {(int(r["season"]), int(r["round"])): r.get("SprintResults", []) for r in _pages(f"{year}/sprint")}
    rows = []
    for race in res:
        key = (int(race["season"]), int(race["round"]))
        qd = {x["Driver"]["driverId"]: x for x in q.get(key, [])}
        sd = {x["Driver"]["driverId"]: x for x in sp.get(key, [])}
        for x in race["Results"]:
            d = x["Driver"]["driverId"]
            qq, ss = qd.get(d, {}), sd.get(d, {})
            rows.append(dict(
                season=key[0], round=key[1], date=race["date"], race=race["raceName"],
                circuit=race["Circuit"]["circuitId"], driver=d, code=x["Driver"].get("code"), team=x["Constructor"]["constructorId"],
                grid=int(x["grid"]), finish=int(x["position"]) if x["positionText"].isdigit() else None,
                order=int(x["position"]), status=x["status"], points=float(x["points"]), laps=int(x["laps"]),
                q_pos=_num(qq.get("position")),
                q1=_qsec(qq.get("Q1")), q2=_qsec(qq.get("Q2")), q3=_qsec(qq.get("Q3")),
                s_grid=_num(ss.get("grid")), s_order=_num(ss.get("position")), s_points=_num(ss.get("points")),
            ))
    return rows


def load_results(first=2018, last=2026, refresh=False):
    f = DATA / "results.csv"
    if f.exists() and not refresh:
        return pd.read_csv(f, parse_dates=["date"])
    rows = []
    for y in range(first, last + 1):
        rows += fetch_season(y)
    df = pd.DataFrame(rows).sort_values(["season", "round", "order"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df.to_csv(f, index=False)
    return df


if __name__ == "__main__":
    df = load_results(refresh=True)
    print(df.groupby("season").agg(rounds=("round", "nunique"), rows=("driver", "size")))
