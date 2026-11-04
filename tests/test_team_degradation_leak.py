"""Target-race degradation cannot change its own pre-weekend team forecast."""

import pandas as pd

from experiments.a3_team_degradation import PART, prior_team
from f1.data import load_results


def test_prior_cutoff():
    results = pd.DataFrame([
        dict(season=2020, round=r, date=pd.Timestamp(f"2020-01-0{r}"), team=team)
        for r in (1, 2, 3) for team in ("ferrari", "mclaren")
    ])
    source = pd.DataFrame([
        dict(season=2020, round=r, team=team, relative_deg=value)
        for r, values in ((1, (0.2, -0.2)), (2, (0.4, -0.4)), (3, (0.6, -0.6)))
        for team, value in zip(("ferrari", "mclaren"), values)
    ])
    target = lambda s: prior_team(results, s).query("round == 2").reset_index(drop=True)
    expected = target(source)
    later = source.copy()
    later.loc[later["round"] >= 2, "relative_deg"] = 9
    pd.testing.assert_frame_equal(target(later), expected)
    earlier = source.copy()
    earlier.loc[earlier["round"] == 1, "relative_deg"] += 1
    assert not target(earlier).prior_deg.equals(expected.prior_deg)


def test_cached_source_cutoff():
    results = load_results().query("season <= 2025")
    source = pd.read_csv(PART)
    target = lambda s: prior_team(results, s).query("season == 2024 and round == 12").reset_index(drop=True)
    expected = target(source)
    later = source.copy()
    later.loc[(later.season > 2024) | ((later.season == 2024) & (later["round"] >= 12)), "relative_deg"] += 1
    pd.testing.assert_frame_equal(target(later), expected)
    earlier = source.copy()
    earlier.loc[(earlier.season == 2024) & (earlier["round"] == 11), "relative_deg"] += 1
    assert not target(earlier).prior_deg.equals(expected.prior_deg)


if __name__ == "__main__":
    test_prior_cutoff()
    test_cached_source_cutoff()
    print("team degradation cutoff test passed")
