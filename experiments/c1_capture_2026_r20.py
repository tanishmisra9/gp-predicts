"""Freeze E135's pre-weekend São Paulo podium-market comparator after its cutoff."""

from pathlib import Path

import pandas as pd

from experiments.c1_kalshi import get, pre_cutoff_quote


EVENT = "KXF1RACEPODIUM-BRAZ26"
CUTOFF = pd.Timestamp("2026-11-05 00:00:00", tz="UTC")
FIRST_SESSION = pd.Timestamp("2026-11-06 15:30:00", tz="UTC")
OUTPUT = Path("data/parts/kalshi_prethu_2026_r20.csv")


def capture(now=None, fetch=get):
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if not CUTOFF <= now < FIRST_SESSION:
        raise ValueError("Prospective capture requires cutoff <= now < first session")
    event = fetch(f"/events/{EVENT}", with_nested_markets="true")["event"]
    assert event["event_ticker"] == EVENT
    rows = []
    for market in event["markets"]:
        opened = pd.Timestamp(market["open_time"])
        candles = (fetch(f"/series/{event['series_ticker']}/markets/{market['ticker']}/candlesticks",
                         start_ts=int(opened.timestamp()), end_ts=int(CUTOFF.timestamp()),
                         period_interval=60)["candlesticks"] if opened < CUTOFF else [])
        rows.append(dict(event=EVENT, ticker=market["ticker"], driver=market["yes_sub_title"],
                         open_utc=opened.isoformat(), cutoff_utc=CUTOFF.isoformat(),
                         **pre_cutoff_quote(candles, CUTOFF.timestamp())))
    quotes = pd.DataFrame(rows)
    assert quotes.ticker.is_unique and quotes.driver.is_unique
    valid = quotes[quotes.quoted].copy()
    assert all(pd.Timestamp(t) <= CUTOFF for t in valid.last_bar_utc)
    return quotes, valid.sort_values(["midpoint", "driver"], ascending=[False, True]).head(3)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    quotes, top = capture()
    OUTPUT.parent.mkdir(exist_ok=True)
    with OUTPUT.open("x") as out:
        quotes.to_csv(out, index=False)
    print(f"saved {len(quotes)} contracts, {int(quotes.quoted.sum())} valid quotes to {OUTPUT}")
    print("market unavailable (<3 quotes)" if len(top) < 3 else top[["driver", "midpoint"]].to_string(index=False))


if __name__ == "__main__":
    main()
