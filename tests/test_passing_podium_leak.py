"""E153 past-only passing feature is unchanged by target/later pass labels."""

import pandas as pd

from experiments.b2_clean_passing import PART
from experiments.b2_passing_podium import prior_driver
from f1.data import load_results


def test_synthetic():
    results = pd.DataFrame([
        dict(season=2020, round=r, date=pd.Timestamp(f"2020-01-0{r}"), code=code)
        for r in (1, 2, 3) for code in ("AAA", "BBB")
    ])
    source = pd.DataFrame([
        dict(season=2020, round=r, trailing=code, converted=y)
        for r, values in ((1, (1, 0)), (2, (0, 1)), (3, (1, 1)))
        for code, y in zip(("AAA", "BBB"), values)
    ])
    target = lambda s: prior_driver(results, s).query("round == 2").reset_index(drop=True)
    expected = target(source)
    later = source.copy()
    later.loc[later["round"] >= 2, "converted"] = 1
    pd.testing.assert_frame_equal(target(later), expected)
    earlier = source.copy()
    earlier.loc[(earlier["round"] == 1) & (earlier.trailing == "AAA"), "converted"] = 0
    assert not target(earlier).prior_pass.equals(expected.prior_pass)


def test_saved_source():
    results = load_results().query("season <= 2025")
    source = pd.read_csv(PART)
    target = lambda s: prior_driver(results, s).query("season == 2024 and round == 12").reset_index(drop=True)
    expected = target(source)
    later = source.copy()
    later.loc[(later.season > 2024) | ((later.season == 2024) & (later["round"] >= 12)), "converted"] = 1
    pd.testing.assert_frame_equal(target(later), expected)
    earlier = source.copy()
    earlier.loc[(earlier.season == 2023) & (earlier["round"] == 12), "converted"] = 1
    assert not target(earlier).prior_pass.equals(expected.prior_pass)


if __name__ == "__main__":
    test_synthetic()
    test_saved_source()
    print("passing podium feature cutoff test passed")
