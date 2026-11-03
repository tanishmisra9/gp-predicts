"""Kalshi feature extraction ignores target-weekend and later price bars."""

from copy import deepcopy

import pandas as pd

from experiments.c1_kalshi import PART, ROUND_BY_EVENT, pre_cutoff_quote


def test_cutoff():
    bars = [
        {"end_period_ts": 100, "volume": "2", "yes_bid": {"close": "0.40"}, "yes_ask": {"close": "0.50"}},
        {"end_period_ts": 200, "volume": "99", "yes_bid": {"close": "0.90"}, "yes_ask": {"close": "0.99"}},
    ]
    expected = pre_cutoff_quote(bars, 150)
    changed = deepcopy(bars)
    changed[1]["volume"] = "9999"
    changed[1]["yes_bid"]["close"] = "0.01"
    assert pre_cutoff_quote(changed, 150) == expected
    assert expected["midpoint"] == 0.45
    assert expected["pre_volume"] == 2
    changed[0]["yes_bid"]["close"] = "0.30"
    assert pre_cutoff_quote(changed, 150)["midpoint"] != expected["midpoint"]


def test_live_dollar_candles():
    bars = [
        {"end_period_ts": 200, "volume_fp": "8", "yes_bid": {"close_dollars": "0.90"},
         "yes_ask": {"close_dollars": "0.99"}},
        {"end_period_ts": 100, "volume_fp": "2", "yes_bid": {"close_dollars": "0.40"},
         "yes_ask": {"close_dollars": "0.50"}},
    ]
    quote = pre_cutoff_quote(bars, 150)
    assert quote["midpoint"] == 0.45
    assert quote["pre_volume"] == 2
    assert quote["last_bar_utc"] == "1970-01-01T00:01:40+00:00"


def test_cached_cutoffs():
    quotes = pd.read_csv(PART)
    dates = (pd.read_csv("data/results.csv").query("season == 2025")
             [["round", "date"]].drop_duplicates())
    checked = quotes.merge(dates, on="round", validate="many_to_one")
    assert len(checked) == len(quotes)
    assert (pd.to_datetime(checked.cutoff_utc, utc=True)
            == pd.to_datetime(checked.date, utc=True) - pd.Timedelta(days=3)).all()
    assert all(ROUND_BY_EVENT[e.rsplit("-", 1)[-1]] == r
               for e, r in zip(quotes.event, quotes["round"]))
    quoted = quotes[quotes.quoted]
    assert not quoted.empty
    assert (pd.to_datetime(quoted.last_bar_utc, utc=True)
            <= pd.to_datetime(quoted.cutoff_utc, utc=True)).all()


if __name__ == "__main__":
    test_cutoff()
    test_live_dollar_candles()
    test_cached_cutoffs()
    print("kalshi leak test passed")
