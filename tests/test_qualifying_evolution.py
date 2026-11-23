"""Hand-checkable E266 compound choice, phase quality and source support."""
import numpy as np
import pandas as pd

from experiments.e266_qualifying_evolution import TIMES, clean, measure, source_summary


def lap(driver, seconds, compound="SOFT", number=1):
    return dict(Driver=driver, Compound=compound, LapNumber=number,
                LapTime=pd.Timedelta(seconds=seconds), Sector1Time=pd.Timedelta(seconds=30),
                Sector2Time=pd.Timedelta(seconds=30), Sector3Time=pd.Timedelta(seconds=seconds-60),
                PitInTime=pd.NaT, PitOutTime=pd.NaT, TrackStatus="1", IsAccurate=True, Deleted=False)


def main():
    a = pd.DataFrame([lap(f"d{i}", 100+i) for i in range(6)])
    b = pd.DataFrame([lap(f"d{i}", 99+i*.9) for i in range(6)])
    expected = measure(a, b)
    delta = np.array([100*(99+i*.9-(100+i))/(100+i) for i in range(6)])
    np.testing.assert_allclose(expected.delta_pct, delta)
    np.testing.assert_allclose(expected.evolution_relative, delta-np.median(delta))
    assert np.isclose(expected.evolution_relative.median(), 0)
    extra_a = pd.concat([a, pd.DataFrame([lap("d0", 150, "MEDIUM", 2), lap("d0", 90, "HARD", 3)])])
    extra_b = pd.concat([b, pd.DataFrame([lap("d0", 101, "MEDIUM", 2)])])
    chosen = measure(extra_a, extra_b).iloc[0]
    assert chosen.compound == "SOFT" and chosen.q1_seconds == 100 and chosen.q3_seconds == 99
    extra_b.loc[extra_b.Compound.eq("MEDIUM"), TIMES] = [lap("d0", 99, "MEDIUM")[c] for c in TIMES]
    assert measure(extra_a, extra_b).iloc[0].compound == "MEDIUM"
    pd.testing.assert_frame_equal(measure(a.sample(frac=1, random_state=1), b.iloc[::-1]), expected)
    pd.testing.assert_frame_equal(measure(a.assign(order=99, grid=-1), b.assign(order=0, win=1)), expected)
    assert measure(a, b.iloc[:5]).evolution_relative.isna().all()
    assert measure(None, b).empty and measure(a, None).empty and measure(a.iloc[:0], b).empty
    for name, value in [("TrackStatus", "14"), ("IsAccurate", False), ("Compound", "WET"),
                        ("Deleted", True), ("PitInTime", pd.Timedelta(seconds=3)),
                        ("PitOutTime", pd.Timedelta(seconds=3)), ("LapTime", pd.NaT)]:
        bad = a.copy(); bad.loc[0, name] = value
        assert "d0" not in set(measure(bad, b).code)
    unknown = a.copy(); unknown.loc[0, "Deleted"] = None
    assert "d0" in set(measure(unknown, b).code)
    boundary = a.copy(); boundary.loc[0, "Sector3Time"] += pd.Timedelta(milliseconds=10)
    assert "d0" in set(clean(boundary).Driver), "exact10ms boundary rejected"
    boundary.loc[0, "Sector3Time"] += pd.Timedelta(nanoseconds=1)
    assert "d0" not in set(clean(boundary).Driver)
    short = pd.DataFrame([lap("x", 70)])
    short.loc[0, "Sector3Time"] += pd.Timedelta(milliseconds=10)
    assert len(clean(short)) == 1, "70-second exact10ms boundary rejected"
    try: measure(pd.concat([a, a.iloc[:1]]), b)
    except AssertionError: pass
    else: raise AssertionError("duplicate driver/lap accepted")
    # Fixed calendar denominator includes unobserved events/entries.
    entries = pd.DataFrame([dict(season=2018+i//21, round=i%21+1, code=f"d{j}")
                            for i in range(103) for j in range(10)])
    table = entries.assign(evolution_relative=np.where(entries.code.lt("d6"), 1., np.nan))
    reports = [dict(available=True, paired_drivers=6) for _ in range(103)]
    assert source_summary(reports, table, entries)["source_gate"]
    reports[0]["paired_drivers"] = 5
    table.loc[table.index[0], "evolution_relative"] = np.nan
    assert not source_summary(reports, table, entries)["source_gate"]
    print("E266 phase/compound/max-gain trap/quality/10ms/support/empty/permutation/labels tests PASS")


if __name__ == "__main__":
    main()
