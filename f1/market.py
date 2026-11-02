"""Polymarket 'Driver Podium Finish' markets -> data/market.csv: market-implied P(podium) per driver at a PRE-WEEKEND time (Wednesday 18:00 UTC before the race,
i.e. before Friday practice) and at race-day close. Public APIs, no key. Coverage is thin (20 completed races: 2025 R21,23,24 and 2026), so it is a benchmark, not training data."""
import json
import re
import unicodedata

import pandas as pd
import requests

from f1.data import DATA, load_results

G = "https://gamma-api.polymarket.com/events"; C = "https://clob.polymarket.com/prices-history"
norm = lambda s: re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower())


def price_at(hist, ts):
    p = [x["p"] for x in hist if x["t"] <= ts]
    return p[-1] if p else None


def build(extra_slugs=None):
    found = json.load(open(DATA / "pm_found.json")); res = load_results(); S = requests.Session(); rows = []
    items = [(tuple(map(int, k.split("-"))), v[0]) for k, v in found.items()] + list((extra_slugs or {}).items())
    for (season, rnd), slug in items:
        r = res[(res.season == season) & (res["round"] == rnd)] if season else res.iloc[0:0]
        ev = S.get(G, params={"slug": slug}, timeout=30).json()
        if not ev: continue
        ev = ev[0]; day = pd.Timestamp(slug.split("podium-")[1]).tz_localize("UTC")
        T = int((day - pd.Timedelta(days=4, hours=6)).timestamp()); Tc = int((day + pd.Timedelta(hours=8)).timestamp())   # Wed 18:00 UTC ; race-day 08:00 UTC
        for m in ev["markets"]:
            name = m.get("groupItemTitle") or ""; last = norm(name.split()[-1]) if name else ""
            toks = json.loads(m["clobTokenIds"]); h = S.get(C, params={"market": toks[0], "interval": "max", "fidelity": 60}, timeout=30).json().get("history", [])
            cand = r[r.driver.map(norm).str.endswith(last)] if len(r) and last else r.iloc[0:0]
            rows.append(dict(season=season, round=rnd, slug=slug, name=name, driver=cand.driver.iloc[0] if len(cand) else None,
                             p_pre=price_at(h, T), p_close=price_at(h, Tc), n_pts=len([x for x in h if x["t"] <= T]), moved=len({x["p"] for x in h if x["t"] <= T}) > 1))
    df = pd.DataFrame(rows); df.to_csv(DATA / "market.csv", index=False); return df


if __name__ == "__main__":
    d = build(); print(d.groupby(["season", "round"]).agg(n=("name", "size"), matched=("driver", lambda s: s.notna().sum()), psum=("p_pre", "sum"), moved=("moved", "mean")).round(2).to_string())
