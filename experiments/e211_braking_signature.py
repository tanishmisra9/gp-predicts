"""E211 cache-only many-lap fixed-speed braking transit measurement."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.e190_traffic_feasibility import cached_session,start_exposure
from experiments.e204_warmup import evaluate
from experiments.e191_traffic_repeatability import cluster_interval
from f1.data import ROOT,load_results

COLS=['first_sample','last_sample','start','end','duration','samples']


def braking(car,origin):
    data=car[car.Date.notna()].drop_duplicates('Date',keep='last').sort_values('Date')
    t=(data.Date-origin).dt.total_seconds().to_numpy()
    delta=np.diff(data.Date.to_numpy(dtype='datetime64[ns]').astype('int64'))
    v=data.Speed.to_numpy(float)
    observed=data.Time.notna().to_numpy()&np.isfinite(v)&data.Brake.eq(True).to_numpy()
    candidates=np.flatnonzero((v[:-1]>=250)&(v[1:]<250))
    rows=[]; consumed=-1
    for i in candidates:
        if i<=consumed or not observed[i]:
            continue
        j=i+1
        while j<len(t) and observed[j] and 0<delta[j-1]<=500_000_000 and v[j]<=v[j-1]:
            if v[j]<=150:
                if j-i+1>=4:
                    start=t[i]+(v[i]-250)/(v[i]-v[i+1])*(t[i+1]-t[i])
                    end=t[j-1]+(v[j-1]-150)/(v[j-1]-v[j])*(t[j]-t[j-1])
                    if .4<=end-start<=3:
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
        number=str(g.DriverNumber.iloc[0]); car=cars.get(number)
        if car is None:
            continue
        segments=braking(car,origin)
        for lap in g.itertuples():
            start=lap.LapStartTime.total_seconds(); end=lap.Time.total_seconds()
            kept=segments[segments.first_sample.ge(start)&segments.last_sample.lt(end)].copy()
            parts.append(kept.assign(driver=driver,lap=int(lap.LapNumber)))
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=COLS+['driver','lap'])


def team_table(source,entries):
    assert source.season.eq(2019).all() and entries.season.eq(2019).all()
    assert not source.duplicated(['round','driver','first_sample']).any()
    legal=entries[['season','round','code']].rename(columns={'code':'driver'})
    assert len(source.merge(legal,on=['season','round','driver'],validate='many_to_one'))==len(source)
    assert np.isfinite(source.duration).all()
    lap_values=source.groupby(['round','driver','lap'],sort=True).duration.median().reset_index()
    rows=[]
    for (round_,team),event in entries.groupby(['round','team'],sort=True):
        drivers=sorted(event.code.unique()); values=[]
        for subset in ('full','half0','half1'):
            estimates=[]
            for driver in drivers:
                g=lap_values[lap_values['round'].eq(round_)&lap_values.driver.eq(driver)]
                if subset!='full':
                    g=g[g.lap.mod(2).eq(int(subset[-1]))]
                estimates.append(float(g.duration.mean()) if len(g)>=(10 if subset=='full' else 5) else np.nan)
            values.append(float(np.mean(estimates)) if len(estimates)==2 and np.isfinite(estimates).all() else np.nan)
        rows.append(dict(season=2019,round=int(round_),team=team,response=values[0],half0=values[1],half1=values[2]))
    table=pd.DataFrame(rows)
    for col in ('response','half0','half1'):
        group=table.groupby('round')[col]
        table['centered_'+col]=(table[col]-group.transform('mean')).where(group.transform('count').ge(6))
    return table


def main():
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True); parts=[]; reports=[]
    for round_ in range(1,22):
        session=cached_session(fastf1,2019,round_); laps=pd.DataFrame(session.laps)
        with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('braking measurement HTTP forbidden')):
            cars=fastf1._api.car_data(session.api_path)
        current=source_segments(laps,cars).assign(season=2019,round=round_)
        parts.append(current); reports.append(dict(round=round_,segments=len(current),driver_laps=int(current.groupby(['driver','lap']).ngroups)))
        print('SOURCE',json.dumps(reports[-1]),flush=True)
    source=pd.concat(parts,ignore_index=True)
    source.to_csv(ROOT/'results/e211_braking_segments.csv',index=False)
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
    utility.to_csv(ROOT/'results/e211_pace_utility.csv',index=False)
    result.update(source=reports,segments=len(source))
    table.to_csv(ROOT/'results/e211_team_measurements.csv',index=False)
    forecast.to_csv(ROOT/'results/e211_temporal_forecasts.csv',index=False)
    (ROOT/'results/e211_measurement_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
