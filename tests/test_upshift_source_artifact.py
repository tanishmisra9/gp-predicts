"""Replay native caches using scalar four-record traversal and separate lap masks."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.e190_traffic_feasibility import cached_session
from f1.data import ROOT


def main():
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True)
    saved=pd.read_csv(ROOT/"results/e244_upshift_events.csv",float_precision="round_trip")
    report=json.loads((ROOT/"results/e244_measurement_summary.json").read_text())
    total=0
    for round_ in range(1,22):
        with patch("requests.adapters.HTTPAdapter.send",side_effect=RuntimeError("audit network forbidden")):
            session=cached_session(fastf1,2019,round_)
            laps=pd.DataFrame(session.laps); cars=fastf1._api.car_data(session.api_path)
        base=laps[laps.LapNumber.gt(1)&laps.LapTime.notna()&laps.Time.notna()&laps.LapStartTime.notna()&
                  laps.PitInTime.isna()&laps.PitOutTime.isna()&laps.TrackStatus.astype(str).eq("1")&laps.IsAccurate.fillna(False)].copy()
        base["seconds"]=base.LapTime.dt.total_seconds()
        med={d:float(g.seconds.median()) for d,g in base.groupby("Driver")}
        clean=base[[r.seconds<1.07*med[r.Driver] for r in base.itertuples()]]
        origin=max((c.Date-c.Time).max() for c in cars.values() if c.Time.notna().any()).round("ms")
        rows=[]
        for driver,g in clean.groupby("Driver",sort=True):
            car=cars.get(str(g.DriverNumber.iloc[0]))
            if car is None:continue
            records=list(car[car.Date.notna()].drop_duplicates("Date",keep="last").sort_values("Date").itertuples())
            intervals=[(r.LapStartTime.total_seconds(),r.Time.total_seconds(),int(r.LapNumber)) for r in g.itertuples()]
            for a,b,c,d in zip(records,records[1:],records[2:],records[3:]):
                if (a.nGear,b.nGear,c.nGear,d.nGear)!=(6,6,7,7):continue
                group=(a,b,c,d)
                if not all(pd.notna(r.Time) and pd.notna(r.RPM) and np.isfinite(r.RPM) and r.RPM>0 and
                           pd.notna(r.Speed) and r.Speed>=180 and pd.notna(r.Throttle) and 98<=r.Throttle<=100 and
                           pd.notna(r.Brake) and r.Brake==False for r in group):continue
                if not all(0<(right.Date-left.Date).total_seconds()<=.5 for left,right in zip(group,group[1:])):continue
                if b.RPM<=c.RPM:continue
                first=(a.Date-origin).total_seconds(); last=(d.Date-origin).total_seconds()
                for start,end,lap in intervals:
                    if start<=first and last<end:
                        rows.append(dict(first_sample=first,last_sample=last,duration=b.RPM/1000,
                                         pre_rpm=float(b.RPM),post_rpm=float(c.RPM),driver=driver,lap=lap,season=2019,round=round_))
        expected=pd.DataFrame(rows,columns=saved.columns)
        actual=saved[saved["round"].eq(round_)]
        sort=["driver","lap","first_sample"]
        pd.testing.assert_frame_equal(actual.sort_values(sort).reset_index(drop=True),expected.sort_values(sort).reset_index(drop=True),check_dtype=False,atol=1e-10,rtol=1e-12)
        r=report["source"][round_-1]
        assert r==dict(round=round_,shifts=len(expected),driver_laps=int(expected.groupby(["driver","lap"]).ngroups))
        total+=len(expected);print("audited",round_,len(expected),flush=True)
    assert total==report["shifts"] and not report["winner_fit"]
    print("E244 all21raw-cache scalar traversal/lap cleaning/event values/counts audit PASS")


if __name__=="__main__":main()
