"""Open-Meteo weather per race -> data/weather.csv.
`rain_act` / `temp_act` / `wind_act` are observed race-day weather: oracle only.
`fc3_rain` / `fc3_temp` use rolling three-day leads relative to each valid hour,
not a common publication cutoff. This builder requests them from 2024 onward;
they are NOT certified pre-weekend inputs and the pre-weekend guard rejects them.
"""
import numpy as np
import pandas as pd
import requests
import requests_cache

from f1.data import DATA, load_results

_s = requests_cache.CachedSession(str(DATA / "weather_cache"), expire_after=None)


def _circuits():
    r = _s.get("https://api.jolpi.ca/ergast/f1/circuits.json", params={"limit": 100}, timeout=30).json()["MRData"]["CircuitTable"]["Circuits"]
    return {c["circuitId"]: (float(c["Location"]["lat"]), float(c["Location"]["long"])) for c in r}


def _window(d, col):
    h = pd.to_datetime(pd.Series(d["hourly"]["time"])); v = pd.Series(d["hourly"][col], dtype=float)
    return v[(h.dt.hour >= 12) & (h.dt.hour <= 17)]


def build():
    df = load_results().drop_duplicates(["season", "round"])[["season", "round", "date", "circuit"]]; xy = _circuits(); rows = []
    for _, r in df.iterrows():
        lat, lon = xy.get(r.circuit, (np.nan, np.nan)); day = pd.Timestamp(r.date).date().isoformat(); row = dict(season=r.season, round=r["round"])
        if np.isnan(lat):
            rows.append(row); continue
        a = _s.get("https://archive-api.open-meteo.com/v1/archive", params=dict(latitude=lat, longitude=lon, start_date=day, end_date=day,
                   hourly="precipitation,temperature_2m,wind_speed_10m", timezone="auto"), timeout=60).json()
        if "hourly" in a:
            row.update(rain_act=_window(a, "precipitation").sum(), temp_act=_window(a, "temperature_2m").mean(), wind_act=_window(a, "wind_speed_10m").mean())
        if r.season >= 2024:
            f = _s.get("https://previous-runs-api.open-meteo.com/v1/forecast", params=dict(latitude=lat, longitude=lon, start_date=day, end_date=day,
                       hourly="precipitation_previous_day3,temperature_2m_previous_day3", timezone="auto", models="best_match"), timeout=60).json()
            if "hourly" in f:
                row.update(fc3_rain=_window(f, "precipitation_previous_day3").sum(), fc3_temp=_window(f, "temperature_2m_previous_day3").mean())
        rows.append(row)
    w = pd.DataFrame(rows); w.to_csv(DATA / "weather.csv", index=False); return w


if __name__ == "__main__":
    w = build(); print(w.describe().round(2).T[["count", "mean", "max"]])
