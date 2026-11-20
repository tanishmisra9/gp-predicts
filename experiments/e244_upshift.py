"""E244: native full-throttle sixth-to-seventh sampled RPM signature."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.e190_traffic_feasibility import cached_session,start_exposure
from experiments.e211_braking_signature import team_table
from experiments.e204_warmup import evaluate
from f1.data import ROOT,load_results

COLS=["first_sample","last_sample","duration","pre_rpm","post_rpm"]


def shifts(car,origin):
    d=car[car.Date.notna()].drop_duplicates("Date",keep="last").sort_values("Date")
    t=(d.Date-origin).dt.total_seconds().to_numpy()
    ns=d.Date.to_numpy(dtype="datetime64[ns]").astype("int64")
    rpm=d.RPM.to_numpy(float); gear=d.nGear.to_numpy(float)
    valid=d.Time.notna().to_numpy()&d.Throttle.between(98,100).to_numpy()&d.Brake.eq(False).to_numpy()
    valid&=d.Speed.ge(180).to_numpy()&np.isfinite(rpm)&(rpm>0)
    rows=[]
    for i in np.flatnonzero((gear[:-1]==6)&(gear[1:]==7)):
        if i<1 or i+2>=len(d):continue
        ids=np.arange(i-1,i+3)
        if not valid[ids].all() or gear[i-1]!=6 or gear[i+2]!=7:continue
        gaps=np.diff(ns[ids])
        if not ((gaps>0)&(gaps<=500_000_000)).all() or rpm[i]<=rpm[i+1]:continue
        rows.append(dict(first_sample=t[i-1],last_sample=t[i+2],duration=rpm[i]/1000,
                         pre_rpm=rpm[i],post_rpm=rpm[i+1]))
    return pd.DataFrame(rows,columns=COLS).astype({"duration":float})


def source_segments(laps,cars):
    clean=start_exposure(laps)
    assert not clean.duplicated(["Driver","LapNumber"]).any()
    origin=max((c.Date-c.Time).max() for c in cars.values() if c.Time.notna().any()).round("ms")
    rows=[]
    for driver,g in clean.groupby("Driver",sort=True):
        assert g.DriverNumber.nunique()==1
        car=cars.get(str(g.DriverNumber.iloc[0]))
        if car is None:continue
        events=shifts(car,origin)
        for lap in g.itertuples():
            kept=events[events.first_sample.ge(lap.LapStartTime.total_seconds())&events.last_sample.lt(lap.Time.total_seconds())]
            rows.append(kept.assign(driver=driver,lap=int(lap.LapNumber)))
    return (pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(columns=COLS+["driver","lap"])).astype({"duration":float})


def main():
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True); parts=[]; reports=[]
    for round_ in range(1,22):
        with patch("requests.adapters.HTTPAdapter.send",side_effect=RuntimeError("upshift HTTP forbidden")):
            session=cached_session(fastf1,2019,round_)
            cars=fastf1._api.car_data(session.api_path)
            current=source_segments(pd.DataFrame(session.laps),cars).assign(season=2019,round=round_)
        parts.append(current); reports.append(dict(round=round_,shifts=len(current),driver_laps=int(current.groupby(["driver","lap"]).ngroups)))
        print("SOURCE",json.dumps(reports[-1]),flush=True)
    source=pd.concat(parts,ignore_index=True)
    source.to_csv(ROOT/"results/e244_upshift_events.csv",index=False)
    entries=load_results().loc[lambda d:d.season.eq(2019),["season","round","code","team"]]
    table=team_table(source,entries)
    table.to_csv(ROOT/"results/e244_team_measurements.csv",index=False)
    summary,forecast=evaluate(table)
    summary.update(source=reports,shifts=len(source),units="thousands of recorded pre-shift RPM; duration is legacy aggregation field",winner_fit=False)
    forecast.to_csv(ROOT/"results/e244_temporal_forecasts.csv",index=False)
    (ROOT/"results/e244_measurement_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("RESULT",json.dumps(summary),flush=True)


if __name__=="__main__":main()
