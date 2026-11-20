import numpy as np
import pandas as pd
from experiments.e258_gear_ratio import measure, windows, gate


def fixture():
    origin = pd.Timestamp("2019-01-01")
    t = np.arange(120)*.25
    car = pd.DataFrame(dict(Date=origin+pd.to_timedelta(t, unit="s"), Time=pd.to_timedelta(t, unit="s"),
                           nGear=6, RPM=10000., Speed=250., Throttle=100, Brake=False))
    laps = pd.DataFrame(dict(LapStartTime=[pd.Timedelta(seconds=-1)], Time=[pd.Timedelta(seconds=40)], LapNumber=[2]))
    return origin, car, laps


def main():
    origin, car, laps = fixture()
    a, e = measure(car, laps, origin)
    assert a == dict(native_samples=120, eligible=118, supported_gears=1, marked=0, episodes=0) and e.empty
    car.loc[60:64, "RPM"] = 12000
    a, e = measure(car, laps, origin)
    assert a["marked"] == 5 and a["episodes"] == 1
    r = e.iloc[0]
    assert r.samples == 5 and r["last"]-r["first"] == 1 and r.first_index == 60 and r.last_index == 64
    for col, val in [("Time", pd.NaT), ("Throttle", 97), ("Throttle", 101), ("RPM", 0), ("RPM", np.nan),
                     ("Speed", 179), ("Brake", True), ("Brake", None), ("nGear", 7)]:
        c = car.copy()
        if val is None: c[col] = c[col].astype(object)
        c.loc[62, col] = val
        assert measure(c, laps, origin)[1].empty, (col, val)
    c = car.copy(); c.loc[62, "RPM"] = 8000
    assert measure(c, laps, origin)[1].empty
    c = car.copy(); c.loc[64, "RPM"] = 10000
    assert measure(c, laps, origin)[1].empty
    c = car.copy(); c.Date = origin+pd.to_timedelta(np.arange(120)*.249, unit="s")
    assert measure(c, laps, origin)[1].empty
    c = car.copy(); c.loc[60:64, "RPM"] = 11000
    assert measure(c, laps, origin)[1].empty
    c = car.copy(); c.loc[60:64, "RPM"] = 11000.01
    assert len(measure(c, laps, origin)[1]) == 1
    c = car.copy(); c.loc[63:, "Date"] += pd.Timedelta(seconds=.250001)
    assert measure(c, laps, origin)[1].empty
    assert measure(car.iloc[:0], laps, origin)[1].empty
    assert measure(car.iloc[:1], laps, origin)[1].empty
    assert measure(car, laps.iloc[:0], origin)[1].empty
    pd.testing.assert_frame_equal(measure(pd.concat([car, car]).sort_index(kind="stable"), laps, origin)[1], e)
    pd.testing.assert_frame_equal(measure(car.assign(order=99, win=1, grid=1, DRS=99), laps, origin)[1], e)
    boundary = laps.copy(); boundary.Time = pd.to_timedelta([16.25], unit="s")
    assert measure(car, boundary, origin)[1].empty
    # The source gate can pass and each required support dimension can fail.
    rows = [dict(season=2019, round=r, driver=str(i), eligible=100, supported_gears=2)
            for r in (5,12,14) for i in range(10)]
    support = pd.DataFrame(rows)
    episodes = pd.DataFrame(dict(season=[2019]*3, round=[5,12,12], driver=["a","b","c"]))
    reports = [dict(available=True)]*3
    assert gate(support, episodes, reports)
    assert not gate(support.assign(supported_gears=1), episodes, reports)
    assert not gate(support, episodes.assign(driver="a"), reports)
    assert not gate(support, episodes.assign(round=5), reports)
    raw = laps.assign(Driver="AAA", LapTime=pd.to_timedelta([100],unit="s"), PitInTime=pd.NaT,
                      PitOutTime=pd.NaT, TrackStatus="1", IsAccurate=False)
    assert len(windows(raw)) == 1  # slow/inaccurate completed lap is retained
    try:
        windows(pd.concat([raw,raw]))
    except AssertionError:
        pass
    else:
        raise AssertionError("duplicate lap accepted")
    print("E258 source sampling/boundaries/invalid/empty/duplicate/outcome/gate tests PASS")


if __name__ == "__main__": main()
