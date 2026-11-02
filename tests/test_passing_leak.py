"""B2 passing forecast cannot use target-race or later pass events."""

import pandas as pd

from experiments.b2_passing import PART, prior_rate
from f1.data import load_results


def test_synthetic_history():
    races = pd.DataFrame([(2022, 1, 0, "AAA"), (2022, 2, 1, "AAA"),
                          (2022, 3, 2, "AAA")], columns=["season", "round", "k", "code"])
    events = races[["season", "round"]].assign(Driver="AAA", passed=[1, 0, 1])
    base = prior_rate(events, races)
    changed = events.copy()
    changed.loc[changed["round"] >= 2, "passed"] = 1
    got = prior_rate(changed, races)
    assert base.loc[1, "pass_rate"] == got.loc[1, "pass_rate"] == 3 / 21
    changed.loc[changed["round"] == 1, "passed"] = 0
    got = prior_rate(changed, races)
    assert base.loc[1, "pass_rate"] != got.loc[1, "pass_rate"]


def test_cached_events_ignore_target_and_later():
    races = load_results()[["season", "round", "code"]].drop_duplicates()
    races = races[races.season <= 2025].copy()
    races["k"] = races.groupby(["season", "round"], sort=True).ngroup()
    events = pd.read_csv(PART)
    base = prior_rate(events, races)
    for year, round_ in ((2023, 7), (2025, 20)):
        changed = events.copy()
        later = (changed.season > year) | ((changed.season == year) & (changed["round"] >= round_))
        changed.loc[later, "passed"] = 1 - changed.loc[later, "passed"]
        got = prior_rate(changed, races)
        target = lambda d: d[(d.season == year) & (d["round"] == round_)].sort_values("code")[["pass_rate", "pass_n"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(target(base), target(got), check_exact=True)


if __name__ == "__main__":
    test_synthetic_history()
    test_cached_events_ignore_target_and_later()
    print("passing leak test passed")
