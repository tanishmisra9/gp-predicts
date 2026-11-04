"""The 2025 target-session profile cannot enter its 2024 team-speed feature."""

import pandas as pd

from experiments.a1_2025_team_transfer import PART, prior_team_speed
from f1.data import DATA, load_results


def test_source_year_cutoff():
    for season in (2023, 2024):
        source = pd.DataFrame([
            dict(circuit="jeddah", season=season, half=half, bin=bin_, code=code, speed=speed)
            for half in (0, 1) for bin_ in (0, 1)
            for code, speed in (("AAA", 100.0), ("BBB", 90.0))
        ])
        results = pd.DataFrame([
            dict(season=season, circuit="jeddah", code="AAA", team="first"),
            dict(season=season, circuit="jeddah", code="BBB", team="second"),
        ])
        expected = prior_team_speed(source, results, season)
        later = source.assign(season=season + 1, speed=999.0)
        pd.testing.assert_frame_equal(prior_team_speed(pd.concat([source, later]), results, season), expected)
        changed = source.copy()
        changed.loc[changed.code.eq("AAA"), "speed"] += 10
        assert not prior_team_speed(changed, results, season).prior_xy_speed.equals(expected.prior_xy_speed)


def test_cached_source_cutoff():
    results = load_results().query("season <= 2025")
    for part in (PART, DATA / "parts/race_xy_multistint_all_2024.csv",
                 DATA / "parts/race_xy_multistint_all_2023.csv"):
        source = pd.read_csv(part)
        year = int(source.season.iloc[0])
        assert source.season.eq(year).all()
        expected = prior_team_speed(source, results, year)
        later = source.assign(season=year + 1, speed=source.speed + 100)
        pd.testing.assert_frame_equal(prior_team_speed(pd.concat([source, later]), results, year), expected)
        changed = source.copy()
        changed.loc[changed.circuit.eq("jeddah") & changed.code.eq("VER"), "speed"] += 10
        assert not prior_team_speed(changed, results, year).prior_xy_speed.equals(expected.prior_xy_speed)


if __name__ == "__main__":
    test_source_year_cutoff()
    test_cached_source_cutoff()
    print("2025 profile cutoff test passed")
