"""Explicit win history must ignore target-and-later labels and react to earlier labels."""
import numpy as np
import pandas as pd

from experiments.e171_win_history import WIN_FEATURES, add_win_history


def check_source(X, k):
    before = add_win_history(X).loc[X.k.eq(k), WIN_FEATURES]
    bad = X.copy(); bad.loc[bad.k.ge(k), "win"] = 1 - bad.loc[bad.k.ge(k), "win"]
    after = add_win_history(bad).loc[X.k.eq(k), WIN_FEATURES]
    np.testing.assert_array_equal(before, after)
    older = X.copy(); older.loc[older.k.eq(k - 1), "win"] = 1 - older.loc[older.k.eq(k - 1), "win"]
    changed = add_win_history(older).loc[X.k.eq(k), WIN_FEATURES]
    assert not np.array_equal(before.to_numpy(), changed.to_numpy())


def test_history_leak():
    X = pd.DataFrame([dict(k=k, driver=d, team=d, era=1, win=int(d == "A"))
                      for k in range(4) for d in ("A", "B")])
    check_source(X, 2)
    first = add_win_history(X).loc[X.k.eq(0), WIN_FEATURES]
    np.testing.assert_allclose(first, [[0.05, 0.05, 0.10, 0.10]] * 2)
    measured = add_win_history(X)
    assert measured.loc[2, "prior_driver_win3"] > measured.loc[3, "prior_driver_win3"]


if __name__ == "__main__":
    test_history_leak()
    source = pd.read_csv("data/results.csv").loc[lambda d: d.season <= 2025].copy()
    source["k"] = source.groupby(["season", "round"], sort=True).ngroup()
    source["era"] = np.where(source.season <= 2021, 0, 1)
    source["win"] = source.order.eq(1).astype(int)
    k = source.loc[source.season.eq(2024) & source["round"].eq(7), "k"].iloc[0]
    check_source(source, k)
    print("win-history target/future scramble, past sensitivity, and no-history checks passed")
