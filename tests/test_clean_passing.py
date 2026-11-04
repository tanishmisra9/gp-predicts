"""E152 filters and past-only conversion rates."""

import pandas as pd

from experiments.b2_clean_passing import PART, forecast, opportunities
from f1.data import load_results


def sample():
    rows = []
    for code, first, second, times in (
        ("AAA", 3, 4, (100, 204, 304)),
        ("BBB", 4, 3, (101, 200, 300)),
    ):
        for lap, time in enumerate(times, 1):
            rows.append(dict(Driver=code, LapNumber=lap,
                             Position=first if lap == 1 else second,
                             Time=pd.Timedelta(seconds=time), TrackStatus="1", IsAccurate=True,
                             PitInTime=pd.NaT, PitOutTime=pd.NaT))
    return pd.DataFrame(rows)


def test_opportunity_filters():
    laps = sample()
    found = opportunities(laps)
    assert len(found) == 1 and found.iloc[0].trailing == "BBB" and found.iloc[0].converted == 1
    far = laps.copy()
    far.loc[(far.Driver == "BBB") & (far.LapNumber == 1), "Time"] = pd.Timedelta(seconds=103)
    assert opportunities(far).empty
    pit = laps.copy()
    pit.loc[(pit.Driver == "AAA") & (pit.LapNumber == 2), "PitInTime"] = pd.Timedelta(seconds=200)
    assert opportunities(pit).empty
    caution = laps.copy()
    caution.loc[(caution.Driver == "AAA") & (caution.LapNumber == 2), "TrackStatus"] = "4"
    assert opportunities(caution).empty


def test_forecast_cutoff():
    results = pd.DataFrame([dict(season=2020, round=r, date=pd.Timestamp(f"2020-01-0{r}")) for r in (1, 2, 3)])
    source = pd.DataFrame([dict(season=2020, round=r, lap=1, trailing="BBB", front="AAA", converted=y)
                           for r, y in ((1, 1), (2, 0), (3, 1))])
    target = lambda s: forecast(results, s).query("round == 2").reset_index(drop=True)
    expected = target(source)
    later = source.copy()
    later.loc[later["round"] >= 2, "converted"] = 1
    pd.testing.assert_frame_equal(target(later).drop(columns="converted"), expected.drop(columns="converted"))
    earlier = source.copy()
    earlier.loc[earlier["round"] == 1, "converted"] = 0
    assert target(earlier).driver_prob.iloc[0] != expected.driver_prob.iloc[0]


def test_saved_source_cutoff():
    results = load_results().query("season <= 2025")
    source = pd.read_csv(PART)
    target = lambda s: forecast(results, s).query("season == 2024 and round == 12").drop(columns="converted").reset_index(drop=True)
    expected = target(source)
    later = source.copy()
    later.loc[(later.season > 2024) | ((later.season == 2024) & (later["round"] >= 12)), "converted"] = 1
    pd.testing.assert_frame_equal(target(later), expected)
    earlier = source.copy()
    earlier.loc[(earlier.season == 2023) & (earlier["round"] == 12), "converted"] = 1
    assert not target(earlier).driver_prob.equals(expected.driver_prob)


if __name__ == "__main__":
    test_opportunity_filters()
    test_forecast_cutoff()
    test_saved_source_cutoff()
    print("clean passing filters and cutoff test passed")
