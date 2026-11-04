"""E146's future-round captures obey their fixed cutoff and immutable output."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from experiments.c1_capture_2026_remaining import RACES, capture, main


EVENT = "KXF1RACEPODIUM-LASV26"
CUTOFF = pd.Timestamp("2026-11-19 00:00:00", tz="UTC")


def fake(path, **params):
    if path == "/events":
        return dict(events=[dict(event_ticker=EVENT, title="Las Vegas Grand Prix Podium Finishers?")], cursor="")
    if path == f"/events/{EVENT}":
        return dict(event=dict(event_ticker=EVENT, markets=[
            dict(ticker=f"{EVENT}-{driver}", yes_sub_title=driver,
                 open_time="2026-11-18T00:00:00Z")
            for driver in ("Beta", "Alpha", "Gamma")]))
    assert params["end_ts"] == int(CUTOFF.timestamp())
    driver = path.split("/markets/", 1)[1].split("/candlesticks", 1)[0].rsplit("-", 1)[-1]
    bid = {"Alpha": 0.4, "Beta": 0.4, "Gamma": 0.2}[driver]
    return dict(candlesticks=[
        dict(end_period_ts=int((CUTOFF - pd.Timedelta(hours=1)).timestamp()),
             yes_bid={"close_dollars": str(bid)}, yes_ask={"close_dollars": str(bid + 0.1)},
             volume_fp="1"),
        dict(end_period_ts=int((CUTOFF + pd.Timedelta(hours=1)).timestamp()),
             yes_bid={"close_dollars": "0.99"}, yes_ask={"close_dollars": "1"},
             volume_fp="1"),
    ])


def test_capture():
    try:
        capture(21, CUTOFF - pd.Timedelta(seconds=1), fake)
        raise AssertionError("early capture accepted")
    except ValueError:
        pass
    record = capture(21, CUTOFF, fake)
    assert record["status"] == "available"
    assert [x["driver"] for x in record["top_three"]] == ["Alpha", "Beta", "Gamma"]
    assert all(pd.Timestamp(x["last_bar_utc"]) <= CUTOFF for x in record["quotes"])
    empty = capture(21, CUTOFF, lambda path, **params: dict(events=[], cursor=""))
    assert empty["status"] == "event_unavailable" and not empty["top_three"]


def test_all_round_guards():
    no_event = lambda path, **params: dict(events=[], cursor="")
    for round_, (_, cutoff_text, first_text) in RACES.items():
        cutoff = pd.Timestamp(cutoff_text, tz="UTC")
        first = pd.Timestamp(first_text, tz="UTC")
        for bad in (cutoff - pd.Timedelta(seconds=1), first):
            try:
                capture(round_, bad, no_event)
                raise AssertionError(f"round {round_} accepted time outside window")
            except ValueError:
                pass
        assert capture(round_, cutoff, no_event)["status"] == "event_unavailable"


def test_immutable_output():
    cwd = Path.cwd()
    with TemporaryDirectory() as tmp:
        try:
            os.chdir(tmp)
            main(21, CUTOFF, fake)
            saved = Path("data/parts/kalshi_prethu_2026_r21.json")
            assert json.loads(saved.read_text())["status"] == "available"
            try:
                main(21, CUTOFF, fake)
                raise AssertionError("existing capture overwritten")
            except FileExistsError:
                pass
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    test_capture()
    test_all_round_guards()
    test_immutable_output()
    print("remaining Kalshi capture tests passed")
