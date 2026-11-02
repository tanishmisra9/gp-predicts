"""B3 strategy exposure uses only completed earlier races."""

import pandas as pd

from experiments.b3_strategy_exposure import prior_exposure, source_table
from experiments.common import table
from f1.data import DATA, load_results


def test_synthetic_history():
    X = pd.DataFrame([(2022, r, r - 1, "AAA", "TEAM", "track") for r in (1, 2, 3)],
                     columns=["season", "round", "k", "code", "team", "circuit"])
    team = X[["season", "round", "team", "circuit"]].assign(stop_resid=[0.2, 0.5, -0.3], n=2)
    cost = X[["season", "round", "circuit"]].assign(pit_cost=[20, 30, 25])
    base = prior_exposure(X, team, cost)
    changed_t, changed_c = team.copy(), cost.copy()
    changed_t.loc[changed_t["round"] >= 2, "stop_resid"] = 99
    changed_c.loc[changed_c["round"] >= 2, "pit_cost"] = 99
    got = prior_exposure(X, changed_t, changed_c)
    assert base.loc[1, "strategy_exposure"] == got.loc[1, "strategy_exposure"] == 4
    changed_t.loc[changed_t["round"] == 1, "stop_resid"] = 1
    got = prior_exposure(X, changed_t, changed_c)
    assert base.loc[1, "strategy_exposure"] != got.loc[1, "strategy_exposure"]


def test_cached_source_ignores_target_and_later():
    results = load_results()
    pits = pd.concat([pd.read_csv(f) for f in sorted((DATA / "parts").glob("pits_*.csv"))], ignore_index=True)
    team, cost = source_table(results, pits)
    X = table()
    X = X[X.season <= 2025].copy()
    X = X.merge(results[["season", "round", "circuit"]].drop_duplicates(),
                on=["season", "round"], validate="many_to_one")
    base = prior_exposure(X, team, cost)
    year, round_ = 2024, 12
    changed_t, changed_c = team.copy(), cost.copy()
    late_t = (changed_t.season > year) | ((changed_t.season == year) & (changed_t["round"] >= round_))
    late_c = (changed_c.season > year) | ((changed_c.season == year) & (changed_c["round"] >= round_))
    changed_t.loc[late_t, "stop_resid"] = 99
    changed_c.loc[late_c, "pit_cost"] = 99
    got = prior_exposure(X, changed_t, changed_c)
    cols = ["prior_team_stops", "prior_pit_cost", "strategy_exposure"]
    target = lambda d: d[(d.season == year) & (d["round"] == round_)].sort_values("code")[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(target(base), target(got), check_exact=True)


if __name__ == "__main__":
    test_synthetic_history()
    test_cached_source_ignores_target_and_later()
    print("strategy exposure leak test passed")
