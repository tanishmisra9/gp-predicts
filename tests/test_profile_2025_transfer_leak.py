"""The 2025 target-session profile cannot enter its 2024 team-speed feature."""

import pandas as pd

from experiments.a1_2025_team_transfer import PART, prior_team_speed
from f1.data import load_results


def test_source_year_cutoff():
    source = pd.DataFrame([
        dict(circuit="jeddah", season=2024, half=half, bin=bin_, code=code, speed=speed)
        for half in (0, 1) for bin_ in (0, 1)
        for code, speed in (("AAA", 100.0), ("BBB", 90.0))
    ])
    results = pd.DataFrame([
        dict(season=2024, circuit="jeddah", code="AAA", team="first"),
        dict(season=2024, circuit="jeddah", code="BBB", team="second"),
    ])
    expected = prior_team_speed(source, results)
    later = source.assign(season=2025, speed=999.0)
    pd.testing.assert_frame_equal(prior_team_speed(pd.concat([source, later]), results), expected)
    changed = source.copy()
    changed.loc[changed.code.eq("AAA"), "speed"] += 10
    assert not prior_team_speed(changed, results).prior_xy_speed.equals(expected.prior_xy_speed)


def test_cached_source_cutoff():
    source = pd.read_csv(PART)
    results = load_results().query("season <= 2025")
    expected = prior_team_speed(source, results)
    later = source.assign(season=2025, speed=source.speed + 100)
    pd.testing.assert_frame_equal(prior_team_speed(pd.concat([source, later]), results), expected)
    changed = source.copy()
    changed.loc[changed.circuit.eq("jeddah") & changed.code.eq("VER"), "speed"] += 10
    assert not prior_team_speed(changed, results).prior_xy_speed.equals(expected.prior_xy_speed)


if __name__ == "__main__":
    test_source_year_cutoff()
    test_cached_source_cutoff()
    print("2025 profile cutoff test passed")
