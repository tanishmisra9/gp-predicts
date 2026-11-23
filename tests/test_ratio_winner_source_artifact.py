"""Independent all-cache scalar selection and source coverage audit."""
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
from tests.test_gear_ratio_source_artifact import scalar
from experiments.e190_traffic_feasibility import cached_session
from f1.data import ROOT,load_results


def main():
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True)
    reports=json.loads((ROOT/"results/e260_source_status.json").read_text())
    inventory=json.loads((ROOT/"results/e212_source_status.json").read_text())
    source=pd.read_csv(ROOT/"results/e260_observations.csv",float_precision="round_trip")
    official=load_results().loc[lambda d:d.season.between(2018,2022),["season","round","code"]].reset_index(drop=True)
    np.testing.assert_array_equal(source[["season","round","code"]],official)
    assert len(reports)==len(inventory)==103
    mapping={};pilot_rows=[];pilot_events=[]
    for report,receipt in zip(reports,inventory):
        season,round_=receipt["season"],receipt["round"]
        assert (report["season"],report["round"])==(season,round_)
        if not receipt["available"]:
            assert not report["available"] and report["rows"]==report["events"]==[]
            continue
        assert report["available"]
        with patch("requests.adapters.HTTPAdapter.send",side_effect=RuntimeError("all-source audit HTTP forbidden")):
            s=cached_session(fastf1,season,round_);laps=pd.DataFrame(s.laps);cars=fastf1._api.car_data(s.api_path)
        assert not laps.duplicated(["Driver","LapNumber"]).any()
        origin=max((c.Date-c.Time).max() for c in cars.values() if c.Time.notna().any()).round("ms")
        expected=[];events=[]
        for driver,g in laps.groupby("Driver",sort=True):
            intervals=[]
            for r in g.itertuples():
                if r.LapNumber>1 and pd.notna(r.LapTime) and r.LapTime>pd.Timedelta(0) and pd.notna(r.Time) and pd.notna(r.LapStartTime) and pd.isna(r.PitInTime) and pd.isna(r.PitOutTime) and str(r.TrackStatus)=="1":
                    intervals.append((origin+r.LapStartTime,origin+r.Time,int(r.LapNumber)))
            intervals.sort();assert all(a<b for a,b,_ in intervals)
            numbers=g.DriverNumber.dropna().unique();assert len(numbers)==1
            car=cars.get(str(numbers[0]))
            if car is None:
                expected.append(dict(driver=driver,available=False));continue
            counts,found=scalar(car,intervals,origin)
            row=dict(driver=driver,available=True,**counts);expected.append(row)
            events.extend(dict(e,driver=driver) for e in found)
            key=(season,round_,driver)
            assert key not in mapping
            mapping[key]=(counts["eligible"],counts["episodes"]) if counts["eligible"]>=100 and counts["supported_gears"]>=2 else (np.nan,np.nan)
            if season==2019 and round_ in (5,12,14):
                pilot_rows.append(dict(season=season,round=round_,driver=driver,**counts))
                pilot_events.extend(dict(e,season=season,round=round_,driver=driver) for e in found)
        assert expected==report["rows"],(season,round_)
        a=pd.DataFrame(events);b=pd.DataFrame(report["events"])
        pd.testing.assert_frame_equal(a,b,check_dtype=False,atol=2e-6,rtol=1e-12)
        print("audited scalar source",season,round_,len(events),flush=True)
    expected=np.array([mapping.get(tuple(k),(np.nan,np.nan)) for k in official.to_numpy()])
    np.testing.assert_allclose(source[["eligible","episodes"]],expected,atol=0,rtol=0,equal_nan=True)
    summary=json.loads((ROOT/"results/e260_source_summary.json").read_text())
    coverage=source.assign(valid=source.eligible.notna()).groupby("season").valid.mean().to_dict()
    assert summary["available"]==sum(r["available"] for r in reports)==99 and summary["events"]==103
    assert summary["coverage"]=={str(k):v for k,v in coverage.items()} and summary["episodes"]==int(source.episodes.sum())
    assert summary["gate"]==bool(min(coverage.values())>=.8)
    old=pd.read_csv(ROOT/"results/e258_driver_support.csv",float_precision="round_trip")
    pd.testing.assert_frame_equal(pd.DataFrame(pilot_rows)[old.columns].reset_index(drop=True),old,check_dtype=False)
    old=pd.read_csv(ROOT/"results/e258_episodes.csv",float_precision="round_trip")
    pd.testing.assert_frame_equal(pd.DataFrame(pilot_events)[old.columns].reset_index(drop=True),old,check_dtype=False,atol=2e-6,rtol=1e-12)
    print("E260 all99source scalar counters/runs/official mapping/coverage/3pilot parity PASS")


if __name__=="__main__":main()
