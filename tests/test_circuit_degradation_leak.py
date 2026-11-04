"""E150: same-circuit history cannot use target or later degradation."""

import pandas as pd

from experiments.a3_circuit_degradation import circuit_forecast
from experiments.a3_team_degradation import PART
from f1.data import load_results


def test_synthetic():
    results = pd.DataFrame([
        dict(season=2020, round=r, date=pd.Timestamp(f"2020-01-0{r}"), circuit=circuit, team=team)
        for r, circuit in ((1, "a"), (2, "b"), (3, "a")) for team in ("ferrari", "mclaren")
    ])
    source = pd.DataFrame([
        dict(season=2020, round=r, team=team, relative_deg=value)
        for r, values in ((1, (0.2, -0.2)), (2, (0.4, -0.4)), (3, (0.6, -0.6)))
        for team, value in zip(("ferrari", "mclaren"), values)
    ])
    target = lambda s: circuit_forecast(results, s).query("round == 3").reset_index(drop=True)
    expected = target(source)
    later = source.copy()
    later.loc[later["round"] >= 3, "relative_deg"] = 9
    pd.testing.assert_frame_equal(target(later), expected)
    earlier = source.copy()
    earlier.loc[earlier["round"] == 1, "relative_deg"] += 1
    assert not target(earlier).circuit_deg.equals(expected.circuit_deg)


def test_saved_source():
    results = load_results().query("season <= 2025")
    source = pd.read_csv(PART)
    target = lambda s: circuit_forecast(results, s).query("season == 2024 and round == 12").reset_index(drop=True)
    expected = target(source)
    later = source.copy()
    mask = (later.season > 2024) | ((later.season == 2024) & (later["round"] >= 12))
    later.loc[mask, "relative_deg"] += 1
    pd.testing.assert_frame_equal(target(later), expected)
    earlier = source.copy()
    prior_round = int(results[(results.season == 2023) & (results.circuit == "silverstone")]["round"].iloc[0])
    earlier.loc[(earlier.season == 2023) & (earlier["round"] == prior_round) & (earlier.team == "ferrari"), "relative_deg"] += 1
    assert not target(earlier).circuit_deg.equals(expected.circuit_deg)


if __name__ == "__main__":
    test_synthetic()
    test_saved_source()
    print("same-circuit degradation cutoff test passed")
