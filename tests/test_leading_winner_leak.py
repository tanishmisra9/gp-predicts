"""Past leading history must ignore all target-and-later race timing."""
import numpy as np
import pandas as pd

from experiments.e179_leading_winner import LEAD_FEATURES, START_FILES, add_leading_history
from f1.data import load_results


def test_leading_history():
    X = pd.DataFrame([dict(season=2022, round=k + 1, k=k, code=c, team="A", era=1)
                      for k in range(3) for c in ("AAA", "BBB")])
    events = X[["season", "round", "code"]].assign(led_share=[.8, .2, .3, .1, .9, .1])
    before = add_leading_history(X, events)
    assert before.loc[X.k.eq(0), LEAD_FEATURES].isna().all().all()
    np.testing.assert_allclose(before.loc[X.k.eq(1), "prior_team_lead3"], 1.)
    np.testing.assert_allclose(before.loc[X.k.eq(1), "prior_driver_lead3"], [.8, .2])
    changed = events.copy(); changed.loc[changed["round"].ge(2), "led_share"] = 99.
    pd.testing.assert_frame_equal(before.loc[X.k.eq(1), LEAD_FEATURES],
                                  add_leading_history(X, changed).loc[X.k.eq(1), LEAD_FEATURES], check_exact=True)
    changed.loc[changed["round"].eq(1), "led_share"] = .1
    assert not before.loc[X.k.eq(1), LEAD_FEATURES].equals(add_leading_history(X, changed).loc[X.k.eq(1), LEAD_FEATURES])
    missing = events.copy(); missing.loc[missing["round"].eq(1), "led_share"] = np.nan
    assert add_leading_history(X, missing).loc[X.k.eq(1), LEAD_FEATURES].isna().all().all()


if __name__ == "__main__":
    test_leading_history()
    X = load_results().loc[lambda d: d.season <= 2025].copy()
    X["k"] = X.groupby(["season", "round"], sort=True).ngroup()
    X["era"] = np.where(X.season <= 2021, 0, 1)
    events = pd.concat([pd.read_csv(p) for p in START_FILES], ignore_index=True).loc[lambda d: d.season <= 2025]
    before = add_leading_history(X, events)
    for year, round_ in ((2023, 7), (2025, 20)):
        changed = events.copy()
        late = changed.season.gt(year) | (changed.season.eq(year) & changed["round"].ge(round_))
        changed.loc[late, "led_share"] = 99.
        mask = X.season.eq(year) & X["round"].eq(round_)
        pd.testing.assert_frame_equal(before.loc[mask, LEAD_FEATURES],
                                      add_leading_history(X, changed).loc[mask, LEAD_FEATURES], check_exact=True)
    print("leading history source-cutoff, earlier-input, team sum and missing checks passed")
