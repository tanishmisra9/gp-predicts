"""E232 fixed low-speed powered transit, offline 2019 source qualification."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.e190_traffic_feasibility import cached_session,start_exposure
from experiments.e211_braking_signature import COLS,team_table
from experiments.e204_warmup import evaluate
from experiments.e191_traffic_repeatability import cluster_interval
from f1.data import ROOT,load_results


def acceleration(car,origin):
    data=car[car.Date.notna()].drop_duplicates('Date',keep='last').sort_values('Date')
    t=(data.Date-origin).dt.total_seconds().to_numpy()
    ns=data.Date.to_numpy(dtype='datetime64[ns]').astype('int64')
    v=data.Speed.to_numpy(float)
    good=data.Time.notna().to_numpy()&np.isfinite(v)&data.Brake.eq(False).to_numpy()
    good&=data.Throttle.between(98,100).to_numpy()
    rows=[]; consumed=-1
    for i in np.flatnonzero((v[:-1]<=100)&(v[1:]>100)):
        if i<=consumed or not good[i]:
            continue
        j=i+1
        while j<len(t) and good[j] and 0<ns[j]-ns[j-1]<=500_000_000 and v[j]>=v[j-1]:
            if v[j]>=150:
                if j-i+1>=4:
                    start=t[i]+(100-v[i])/(v[i+1]-v[i])*(t[i+1]-t[i])
                    end=t[j-1]+(150-v[j-1])/(v[j]-v[j-1])*(t[j]-t[j-1])
                    if .4<=end-start<=4:
                        rows.append(dict(first_sample=t[i],last_sample=t[j],start=start,end=end,duration=end-start,samples=j-i+1))
                        consumed=j
                break
            j+=1
    return pd.DataFrame(rows,columns=COLS)


def source_segments(laps,cars):
    clean=start_exposure(laps)
    origin=max((c.Date-c.Time).max() for c in cars.values() if c.Time.notna().any()).round('ms')
    parts=[]
    for driver,g in clean.groupby('Driver',sort=True):
        car=cars.get(str(g.DriverNumber.iloc[0]))
        if car is None:
            continue
        segments=acceleration(car,origin)
        for lap in g.itertuples():
            kept=segments[segments.first_sample.ge(lap.LapStartTime.total_seconds())&segments.last_sample.lt(lap.Time.total_seconds())].copy()
            parts.append(kept.assign(driver=driver,lap=int(lap.LapNumber)))
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=COLS+['driver','lap'])


def main():
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True); parts=[]; reports=[]
    for round_ in range(1,22):
        session=cached_session(fastf1,2019,round_); laps=pd.DataFrame(session.laps)
        with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('acceleration HTTP forbidden')):
            cars=fastf1._api.car_data(session.api_path)
        current=source_segments(laps,cars).assign(season=2019,round=round_)
        parts.append(current); reports.append(dict(round=round_,segments=len(current),driver_laps=int(current.groupby(['driver','lap']).ngroups)))
        print('SOURCE',json.dumps(reports[-1]),flush=True)
    source=pd.concat(parts,ignore_index=True)
    source.to_csv(ROOT/'results/e232_acceleration_segments.csv',index=False)
    entries=load_results().loc[lambda d:d.season.eq(2019),['season','round','code','team']]
    table=team_table(source,entries); result,forecast=evaluate(table)
    errors=pd.read_csv(ROOT/'results/e194_forecasts_errors_scales.csv').loc[lambda d:d.season.eq(2019)]
    errors=errors.groupby(['round','team'],sort=True).error.mean().rename('pace_error').reset_index()
    field=errors.groupby('round').pace_error
    errors['pace_error']=(errors.pace_error-field.transform('mean')).where(field.transform('count').ge(6))
    utility=forecast.merge(errors,on=['round','team'],how='left',validate='one_to_one')
    utility=utility[utility.past_count.ge(3)&utility.pace_error.notna()]
    utility_gate=False
    result.update(utility_rows=len(utility),utility_races=int(utility['round'].nunique()))
    if result['coverage_gate'] and len(utility)>=60 and utility['round'].nunique()>=12:
        correlation,ci=cluster_interval(utility,'prediction','pace_error')
        utility_gate=bool(correlation>=.15 and ci[0]>0)
        result.update(pace_residual_correlation=correlation,pace_residual_ci=ci.tolist())
    result['signature_gate']=result['gate']; result['utility_gate']=utility_gate
    result['gate']=bool(result['signature_gate'] and utility_gate)
    utility.to_csv(ROOT/'results/e232_pace_utility.csv',index=False)
    result.update(source=reports,segments=len(source))
    table.to_csv(ROOT/'results/e232_team_measurements.csv',index=False)
    forecast.to_csv(ROOT/'results/e232_temporal_forecasts.csv',index=False)
    (ROOT/'results/e232_measurement_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
