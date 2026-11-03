"""E132: pre-weekend Kalshi podium-market coverage and 2025 ranking check."""

import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.common import table


BASE = "https://external-api.kalshi.com/trade-api/v2"
PART = "data/parts/kalshi_prethu_2025.csv"
ROUND_BY_EVENT = {
    "JGP25": 3, "BGP25": 4, "SAGP25": 5, "MGP25": 6, "ERGP25": 7,
    "MCGP25": 8, "SGP25": 9, "CAGP25": 10, "AUGP25": 11, "BRGP25": 12,
    "BELGP25": 13, "HUNGP25": 14, "DUTGP25": 15, "ITAGP25": 16,
    "AZEGP25": 17, "SINGP25": 18, "UNISGP25": 19, "GRAPDM25": 20,
    "SAOPGP25": 21, "LASVGP25": 22, "QATGP25": 23, "ABUDGP25": 24,
}


def get(path, **params):
    url = BASE + path + ("?" + urlencode(params) if params else "")
    for attempt in range(5):
        try:
            with urlopen(url, timeout=20) as response:
                return json.load(response)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def pre_cutoff_quote(candles, cutoff):
    before = [c for c in candles if c["end_period_ts"] <= cutoff]
    last = max(before, key=lambda c: c["end_period_ts"]) if before else {}
    bid = last.get("yes_bid", {}).get("close_dollars", last.get("yes_bid", {}).get("close"))
    ask = last.get("yes_ask", {}).get("close_dollars", last.get("yes_ask", {}).get("close"))
    bid = float(bid) if bid is not None else np.nan
    ask = float(ask) if ask is not None else np.nan
    return dict(bid=bid, ask=ask,
                midpoint=(bid + ask) / 2 if np.isfinite(bid + ask) else np.nan,
                pre_volume=sum(float(c.get("volume_fp", c.get("volume", 0))) for c in before),
                quoted=bool(np.isfinite(bid + ask) and 0 <= bid <= ask <= 1),
                last_bar_utc=pd.Timestamp(last["end_period_ts"], unit="s", tz="UTC").isoformat()
                if last else "")


def extract(races):
    events = get("/events", series_ticker="KXF1RACEPODIUM", limit=200)["events"]
    rows = pd.read_csv(PART).to_dict("records") if Path(PART).exists() else []
    done = {row["event"] for row in rows}
    for event in events:
        ticker = event["event_ticker"]
        if not ticker.endswith("25") or ticker in done:
            continue
        markets = get("/historical/markets", event_ticker=ticker, limit=1000)["markets"]
        if not markets:
            continue
        round_ = ROUND_BY_EVENT[ticker.rsplit("-", 1)[-1]]
        race = races[races["round"] == round_].iloc[0]
        cutoff = (pd.Timestamp(race.date, tz="UTC") - pd.Timedelta(days=3)).timestamp()
        for market in markets:
            start = int(pd.Timestamp(market["open_time"]).timestamp())
            candles = (get(f"/historical/markets/{market['ticker']}/candlesticks",
                           start_ts=start, end_ts=int(cutoff), period_interval=60).get("candlesticks", [])
                       if start < cutoff else [])
            rows.append(dict(season=2025, round=int(race["round"]), race=race.race,
                             event=ticker, code=market["ticker"].rsplit("-", 1)[-1],
                             cutoff_utc=pd.Timestamp(cutoff, unit="s", tz="UTC").isoformat(),
                             open_utc=market["open_time"],
                             **pre_cutoff_quote(candles, cutoff)))
        pd.DataFrame(rows).to_csv(PART, index=False)
        print(ticker, race["round"], sum(x["quoted"] for x in rows if x["event"] == ticker), flush=True)
    return pd.DataFrame(rows)


def main():
    history = table()
    races = (history[history.season == 2025][["round"]].drop_duplicates()
             .merge(pd.read_csv("data/results.csv").query("season == 2025")[["round", "date", "race"]]
                    .drop_duplicates(), on="round").sort_values("round"))
    quotes = extract(races)
    assert not quotes.duplicated(["round", "code"]).any()
    assert quotes.groupby("round").event.nunique().eq(1).all()
    X = history[(history.season == 2025) & history["round"].isin(quotes["round"])].merge(
        quotes[["round", "code", "midpoint", "quoted", "pre_volume"]],
        on=["round", "code"], how="left", validate="one_to_one")
    rows = []
    for round_, g in X.groupby("round"):
        if not g.quoted.fillna(False).any():
            continue
        score = -g.midpoint
        standings = -g.d_season_pts.fillna(0) * 100 + g.d10_order.fillna(11) * 1e-3
        rows.append(dict(round=round_, n=len(g), quoted=int(g.quoted.fillna(False).sum()),
                         traded=int(g.pre_volume.fillna(0).gt(0).sum()), volume=g.pre_volume.sum(),
                         market=W.metrics(g, score, tie=None)["podium"],
                         standings=W.metrics(g, standings, tie=None)["podium"]))
    R = pd.DataFrame(rows)
    def shuffled(tr, te, F, **kw):
        t = tr.copy()
        t["podium"] = np.random.default_rng(0).permutation(t.podium.values)
        return W.m_logit_pod(t, te, F, C=0.03, hl=100)
    W.MODELS["null_shuffled_labels"] = shuffled
    null, _ = W.walk_forward(history, [2025], ["null_shuffled_labels"], tie=None)
    R = R.merge(null[["round", "podium"]].rename(columns={"podium": "null"}), on="round",
                validate="one_to_one")
    print(R.to_string(index=False))
    print("strict", R[["market", "standings", "null"]].mean().to_dict(), "races", len(R))
    print("paired", W.paired_boot(R.market, R.standings))
    q = quotes[quotes.quoted].copy()
    q["spread"] = q.ask - q.bid
    print("coverage", len(R), "of", len(quotes.groupby("round")),
          "quotes", len(q), "precutoff traded", int(q.pre_volume.gt(0).sum()),
          "median spread", q.spread.median(), "max spread", q.spread.max())


if __name__ == "__main__":
    main()
