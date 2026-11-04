"""E148: prospective cutoff, source-time, and immutable-output guards."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from experiments.c1_capture_manifold_2026 import RACES, capture, main


class Response:
    def __init__(self, value):
        self.value = value

    def raise_for_status(self):
        pass

    def json(self):
        return self.value


class Session:
    def __init__(self, cutoff):
        self.cutoff = int(cutoff.timestamp() * 1000)

    def get(self, url, params=None, timeout=None):
        if "search-markets" in url:
            return Response([dict(id="m", question="2026 São Paulo Grand Prix podium finishers?",
                                  outcomeType="MULTIPLE_CHOICE", createdTime=self.cutoff - 100)])
        if url.endswith("market/m"):
            return Response(dict(shouldAnswersSumToOne=False, answers=[
                dict(id=d, text=n, createdTime=self.cutoff - 100)
                for d, n in (("a", "Alpha Driver"), ("b", "Beta Racer"), ("c", "Gamma Pilot"))]))
        assert params["beforeTime"] == self.cutoff + 1
        return Response([dict(id=d, answerId=d, createdTime=self.cutoff - 20,
                              fills=[dict(timestamp=self.cutoff - 10)], probAfter=p,
                              amount=10, isFilled=True, isCancelled=False)
                         for d, p in (("a", 0.5), ("b", 0.7), ("c", 0.3))] +
                        [dict(id="later", answerId="c", createdTime=self.cutoff + 1,
                              fills=[dict(timestamp=self.cutoff + 1)], probAfter=0.99,
                              amount=100, isFilled=True, isCancelled=False)])


def main_test():
    cutoff = pd.Timestamp("2026-11-05 00:00:00", tz="UTC")
    entrants = {"alpha_driver", "beta_racer", "gamma_pilot"}
    for r, (_, _, cutoff_text, first_text) in RACES.items():
        c = pd.Timestamp(cutoff_text, tz="UTC")
        f = pd.Timestamp(first_text, tz="UTC")
        for bad in (c - pd.Timedelta(seconds=1), f):
            try:
                capture(r, bad, Session(c), entrants)
                raise AssertionError(f"round {r} accepted invalid time")
            except ValueError:
                pass
    record = capture(20, cutoff, Session(cutoff), entrants)
    assert record["status"] == "available"
    assert record["top_three"] == ["beta_racer", "alpha_driver", "gamma_pilot"]
    assert all(q["betTime"] <= int(cutoff.timestamp() * 1000)
               and q["lastFillTime"] <= int(cutoff.timestamp() * 1000)
               for q in record["selected"]["quotes"].values())
    cwd = Path.cwd()
    with TemporaryDirectory() as tmp:
        try:
            os.chdir(tmp)
            main(20, cutoff, Session(cutoff), entrants)
            assert json.loads(Path("data/parts/manifold_prethu_2026_r20.json").read_text())["status"] == "available"
            try:
                main(20, cutoff, Session(cutoff), entrants)
                raise AssertionError("existing capture overwritten")
            except FileExistsError:
                pass
        finally:
            os.chdir(cwd)
    print("Manifold prospective capture tests passed")


if __name__ == "__main__":
    main_test()
