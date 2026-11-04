"""A first-stop reversal is only counted with the fixed clean timing window."""

import pandas as pd

from experiments.b3_undercut_gate import opportunities


def sample():
    rows = []
    for driver, stop, initial, final in (("AAA", 4, 3, 2), ("BBB", 6, 2, 3)):
        for lap in range(1, 10):
            rows.append(dict(
                Driver=driver, LapNumber=lap, Position=final if lap >= 8 else initial,
                Time=pd.Timedelta(seconds=100 * lap + (1 if driver == "AAA" else 0)),
                PitInTime=pd.Timedelta(seconds=100 * lap) if lap == stop else pd.NaT,
                PitOutTime=pd.Timedelta(seconds=100 * lap) if lap == stop + 1 else pd.NaT,
                TrackStatus="1", IsAccurate=lap not in (stop, stop + 1),
            ))
    return pd.DataFrame(rows)


def test_conversion_and_filters():
    l = sample()
    p = opportunities(l, {"AAA": "first", "BBB": "second"})
    assert len(p) == 1 and p.iloc[0].converted == 1 and p.iloc[0].team == "first"
    slow = l.copy()
    slow.loc[(slow.Driver == "AAA") & slow.LapNumber.eq(3), "Time"] += pd.Timedelta(seconds=2)
    assert opportunities(slow, {}).empty
    caution = l.copy()
    caution.loc[(caution.Driver == "AAA") & caution.LapNumber.eq(5), "TrackStatus"] = "4"
    assert opportunities(caution, {}).empty


if __name__ == "__main__":
    test_conversion_and_filters()
    print("undercut gate tests passed")
