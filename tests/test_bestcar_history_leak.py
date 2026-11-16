"""E177 must summarize past best cars and ignore target/future measurements."""
import numpy as np
import pandas as pd

from experiments.e177_bestcar_winner import BEST_FEATURES, INDICATORS, add_bestcar_history
from f1.data import DATA, load_results
from f1.features import _prep


def test_bestcar_history():
    X = pd.DataFrame(dict(k=[0, 0, 1, 1, 2, 2], team=["A"] * 6, era=[0] * 6))
    source = X.assign(order=[1., 9., 2., 8., 3., 7.], q_gap=[.1, .9, .2, .8, .3, .7], pace_gap=[.2, 1., .4, .9, .5, .8])
    result = add_bestcar_history(X, source)
    assert result.loc[X.k.eq(0), BEST_FEATURES].isna().all().all()
    np.testing.assert_allclose(result.loc[X.k.eq(1), "prior_bestcar3_order"], 1.)
    np.testing.assert_allclose(result.loc[X.k.eq(1), "prior_bestcar3_q_gap"], .1)
    np.testing.assert_allclose(result.loc[X.k.eq(1), "prior_bestcar3_pace_gap"], .2)
    bad = source.copy(); bad.loc[bad.k.ge(1), INDICATORS] = 999.
    changed = add_bestcar_history(X, bad)
    pd.testing.assert_frame_equal(result.loc[X.k.eq(1), BEST_FEATURES], changed.loc[X.k.eq(1), BEST_FEATURES], check_exact=True)
    bad.loc[bad.k.eq(0), INDICATORS] = 50.
    altered = add_bestcar_history(X, bad)
    assert not result.loc[X.k.eq(1), BEST_FEATURES].equals(altered.loc[X.k.eq(1), BEST_FEATURES])


if __name__ == "__main__":
    test_bestcar_history()
    df = load_results().loc[lambda d: d.season <= 2025]
    pace = pd.read_csv(DATA / "racepace_ff1.csv").loc[lambda d: d.season <= 2025]
    source = _prep(df, pace)
    X = source[["season", "round", "k", "team", "era"]].copy()
    baseline = add_bestcar_history(X, source)
    for year, round_ in ((2023, 7), (2025, 20)):
        mask = X.season.eq(year) & X["round"].eq(round_)
        k = X.loc[mask, "k"].iloc[0]
        changed = source.copy(); changed.loc[changed.k.ge(k), INDICATORS] = 999.
        got = add_bestcar_history(X, changed)
        pd.testing.assert_frame_equal(baseline.loc[mask, BEST_FEATURES], got.loc[mask, BEST_FEATURES], check_exact=True)
        changed.loc[changed.k.eq(k - 1), INDICATORS] = 999.
        got = add_bestcar_history(X, changed)
        assert not baseline.loc[mask, BEST_FEATURES].equals(got.loc[mask, BEST_FEATURES])
    print("best-car min, source cutoff, past sensitivity and missing-history checks passed")
