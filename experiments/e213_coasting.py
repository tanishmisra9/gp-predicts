"""E213: observed high-speed pre-braking coasting, fixed 2019 measurement."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.e190_traffic_feasibility import cached_session,start_exposure
from experiments.e204_warmup import evaluate
from experiments.e211_braking_signature import team_table
from experiments.e191_traffic_repeatability import cluster_interval
from f1.data import ROOT,load_results

COLS=['driver','lap','first_sample','duration','coast_seconds','coverage','runs']


def lap_measurement(car,origin,start,end):
    """Return (coast seconds, observed coverage, runs), or None if <90% covered."""
    assert np.isfinite([start,end]).all() and end>start
    data=car[car.Date.notna()].drop_duplicates('Date',keep='last').sort_values('Date')
    t=(data.Date-origin).dt.total_seconds().to_numpy()
    data=data[(t>=start)&(t<end)]
    t=(data.Date-origin).dt.total_seconds().to_numpy()
    dt=np.diff(data.Date.to_numpy(dtype='datetime64[ns]').astype('int64'))/1e9
    v=data.Speed.to_numpy(float); throttle=data.Throttle.to_numpy(float)
    brake=data.Brake
    good=data.Time.notna().to_numpy()&np.isfinite(v)&np.isfinite(throttle)&(v>=0)&(throttle>=0)&(throttle<=100)&brake.isin([True,False]).to_numpy()
    links=(dt>0)&(dt<=.5)&good[:-1]&good[1:]
    coverage=float(dt[links].sum()/(end-start))
    if coverage<.9:
        return None
    coast=good&(v>=200)&(throttle<=5)&brake.eq(False).to_numpy()
    edges=np.diff(np.r_[False,coast,False].astype(int))
    seconds=0.; count=0
    for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        # [a,b) is the maximal observed coast run; b is the first brake sample.
        if a==0 or b==len(data) or b-a<3:
            continue
        if not (good[a-1] and throttle[a-1]>5 and brake.iloc[a-1]==False and v[a-1]>=200):
            continue
        if not (good[b] and brake.iloc[b]==True and v[b]>=200 and links[a-1:b].all()):
            continue
        duration=float(t[b]-t[a])
        if .4<=duration<=3:
            seconds+=duration; count+=1
    return seconds,coverage,count


def source_laps(laps,cars):
    clean=start_exposure(laps)
    origin=max((c.Date-c.Time).max() for c in cars.values() if c.Time.notna().any()).round('ms')
    rows=[]
    for driver,g in clean.groupby('Driver',sort=True):
        car=cars.get(str(g.DriverNumber.iloc[0]))
        if car is None:
            continue
        for lap in g.itertuples():
            start=lap.LapStartTime.total_seconds(); end=lap.Time.total_seconds()
            result=lap_measurement(car,origin,start,end)
            if result is None:
                continue
            seconds,coverage,runs=result
            rows.append(dict(driver=driver,lap=int(lap.LapNumber),first_sample=start,
                duration=100*seconds/(end-start),coast_seconds=seconds,coverage=coverage,runs=runs))
    return pd.DataFrame(rows,columns=COLS)


def main():
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True); parts=[]; reports=[]
    for round_ in range(1,22):
        session=cached_session(fastf1,2019,round_)
        with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('coasting HTTP forbidden')):
            cars=fastf1._api.car_data(session.api_path)
        current=source_laps(pd.DataFrame(session.laps),cars).assign(season=2019,round=round_)
        parts.append(current); reports.append(dict(round=round_,laps=len(current),runs=int(current.runs.sum())))
        print('SOURCE',json.dumps(reports[-1]),flush=True)
    source=pd.concat(parts,ignore_index=True)
    source.to_csv(ROOT/'results/e213_coasting_laps.csv',index=False)
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
    result.update(source=reports,laps=len(source),runs=int(source.runs.sum()))
    table.to_csv(ROOT/'results/e213_team_measurements.csv',index=False)
    forecast.to_csv(ROOT/'results/e213_temporal_forecasts.csv',index=False)
    utility.to_csv(ROOT/'results/e213_pace_utility.csv',index=False)
    (ROOT/'results/e213_measurement_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
