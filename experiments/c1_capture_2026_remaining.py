"""E146: immutable pre-weekend market captures for 2026 rounds 21–23."""

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.c1_kalshi import get, pre_cutoff_quote


RACES = {
    21: ("Las Vegas Grand Prix", "2026-11-19 00:00:00", "2026-11-20 00:30:00"),
    22: ("Qatar Grand Prix", "2026-11-26 00:00:00", "2026-11-27 13:30:00"),
    23: ("Abu Dhabi Grand Prix", "2026-12-03 00:00:00", "2026-12-04 09:30:00"),
}
SERIES = "KXF1RACEPODIUM"


def capture(round_, now=None, fetch=get):
    name, cutoff_text, first_text = RACES[round_]
    cutoff = pd.Timestamp(cutoff_text, tz="UTC")
    first = pd.Timestamp(first_text, tz="UTC")
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if not cutoff <= now < first:
        raise ValueError("Prospective capture requires cutoff <= now < first session")
    events = []
    cursor = ""
    while True:
        page = fetch("/events", series_ticker=SERIES, limit=200, **({"cursor": cursor} if cursor else {}))
        events.extend(page["events"])
        cursor = page.get("cursor", "")
        if not cursor:
            break
    matches = [e for e in events if e["event_ticker"].endswith("26")
               and name.lower() in e["title"].lower()
               and "podium finishers" in e["title"].lower()]
    record = dict(round=round_, race=name, cutoff_utc=cutoff.isoformat(),
                  captured_utc=now.isoformat(), matched_events=[e["event_ticker"] for e in matches],
                  status="event_unavailable", quotes=[], top_three=[])
    if len(matches) != 1:
        return record
    event = fetch(f"/events/{matches[0]['event_ticker']}", with_nested_markets="true")["event"]
    if event["event_ticker"] != matches[0]["event_ticker"]:
        raise ValueError("Event ticker changed between list and detail")
    rows = []
    for market in event["markets"]:
        opened = pd.Timestamp(market["open_time"])
        candles = (fetch(f"/series/{SERIES}/markets/{market['ticker']}/candlesticks",
                         start_ts=int(opened.timestamp()), end_ts=int(cutoff.timestamp()),
                         period_interval=60)["candlesticks"] if opened < cutoff else [])
        rows.append(dict(ticker=market["ticker"], driver=market["yes_sub_title"],
                         open_utc=opened.isoformat(), **pre_cutoff_quote(candles, cutoff.timestamp())))
    quotes = pd.DataFrame(rows)
    if len(quotes):
        if not quotes.ticker.is_unique or not quotes.driver.is_unique:
            raise ValueError("Duplicate market ticker or driver")
        valid = quotes[quotes.quoted].copy()
        if any(pd.Timestamp(t) > cutoff for t in valid.last_bar_utc):
            raise ValueError("Post-cutoff bar in valid quotes")
        top = valid.sort_values(["midpoint", "driver"], ascending=[False, True]).head(3)
        record["quotes"] = json.loads(quotes.to_json(orient="records"))
        record["top_three"] = top[["driver", "midpoint"]].to_dict("records") if len(top) == 3 else []
        record["status"] = "available" if len(top) == 3 else "insufficient_quotes"
    return record


def main(round_, now=None, fetch=get):
    output = Path(f"data/parts/kalshi_prethu_2026_r{round_}.json")
    if output.exists():
        raise FileExistsError(output)
    record = capture(round_, now, fetch)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as out:
        json.dump(record, out, indent=2, allow_nan=False)
        out.write("\n")
    print(output, record["status"], "contracts", len(record["quotes"]),
          "top three", record["top_three"])
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("round", type=int, choices=RACES)
    main(parser.parse_args().round)
