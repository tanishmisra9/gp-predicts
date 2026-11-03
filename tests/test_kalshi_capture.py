"""Prospective capture cannot start early, run after practice, or use late bars."""

import pandas as pd

from experiments.c1_capture_2026_r20 import CUTOFF, FIRST_SESSION, capture


def test_capture():
    for now in (CUTOFF - pd.Timedelta(seconds=1), FIRST_SESSION):
        try:
            capture(now=now, fetch=lambda *a, **k: (_ for _ in ()).throw(AssertionError("fetched")))
        except ValueError:
            pass
        else:
            raise AssertionError("time guard did not reject capture")

    markets = [dict(ticker=f"E-{name}", yes_sub_title=name, open_time="2026-11-03T05:06:00Z")
               for name in ("A", "B", "C", "D")]
    bids = {"A": ".50", "B": ".70", "C": ".70", "D": ".80"}
    def fetch(path, **params):
        if path.startswith("/events/"):
            return {"event": dict(event_ticker="KXF1RACEPODIUM-BRAZ26",
                                  series_ticker="KXF1RACEPODIUM", markets=markets)}
        name = path.rsplit("-", 1)[-1].split("/")[0]
        bar = lambda ts, bid: dict(end_period_ts=ts, volume_fp="2",
                                   yes_bid={"close_dollars": bid}, yes_ask={"close_dollars": ".90"})
        return {"candlesticks": [bar(int(CUTOFF.timestamp()) + 3600, ".89"),
                                bar(int(CUTOFF.timestamp()), bids[name])]}
    quotes, top = capture(now=CUTOFF, fetch=fetch)
    assert len(quotes) == 4 and len(top) == 3
    assert top.driver.tolist() == ["D", "B", "C"]
    assert all(quotes.last_bar_utc == CUTOFF.isoformat())


if __name__ == "__main__":
    test_capture()
    print("kalshi capture test passed")
