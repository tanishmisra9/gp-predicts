"""E203: wind magnitude design support, without any lap-time response fit."""
import json

import numpy as np
import pandas as pd

from experiments.e190_traffic_feasibility import PILOT
from experiments.e198_temperature_support import design,prepare
from f1.data import ROOT


def attach(laps,weather):
    keys=['season','round','weather_time']
    source=weather.rename(columns={'Time':'weather_time'})[keys+['WindSpeed']].copy()
    assert set(map(tuple,source[['season','round']].drop_duplicates().to_numpy()))==set(PILOT)
    source.weather_time=pd.to_timedelta(source.weather_time)
    left=laps.copy(); left.weather_time=pd.to_timedelta(left.weather_time)
    return left.merge(source,on=keys,how='left',validate='many_to_one')


def support(g):
    Z=np.column_stack([design(g),g.TrackTemp.to_numpy(float)/10])
    wind=g.WindSpeed.to_numpy(float)
    if len(g):
        beta,_,rank,_=np.linalg.lstsq(Z,wind,rcond=1e-10)
        residual=wind-Z@beta
        rms=float(np.sqrt(np.mean(residual**2))); variance=float(np.var(wind))
        fraction=float(np.mean(residual**2)/variance) if variance>1e-12 else 0.
    else:
        rank=0; residual=np.array([]); rms=np.nan; fraction=0.
    n=len(g); readings=g.weather_time.nunique(); dof=n-int(rank)
    return dict(rows=n,readings=int(readings),rank=int(rank),dof=dof,residual_rms=rms,
        variance_fraction=fraction,support=bool(n>=15 and readings>=10 and dof>=10 and rms>=.5 and fraction>=.1)),residual


def measure(frame):
    data,excluded=prepare(frame)
    valid=np.isfinite(data.WindSpeed)&data.WindSpeed.ge(0)
    excluded+=int((~valid).sum()); data=data[valid].copy()
    reports=[]; residuals=[]
    for season,round_,driver in frame[['season','round','Driver']].drop_duplicates().itertuples(index=False,name=None):
        g=data[data.season.eq(season)&data['round'].eq(round_)&data.Driver.eq(driver)]
        for name,subset in [('full',g),('half0',g[g.half.eq(0)]),('half1',g[g.half.eq(1)])]:
            values,residual=support(subset)
            reports.append(dict(season=int(season),round=int(round_),driver=driver,subset=name,**values))
            rows=subset[['season','round','Driver','LapNumber','weather_time','WindSpeed','phase','half']].copy()
            rows['subset']=name; rows['residual_wind']=residual; residuals.append(rows)
    table=pd.DataFrame(reports); events=[]
    for season,round_ in PILOT:
        event=table[table.season.eq(season)&table['round'].eq(round_)]
        passed=event.groupby('driver').support.agg(['all','size'])
        events.append(dict(season=season,round=round_,drivers=int(len(passed)),
            passing_drivers=int((passed['all']&passed['size'].eq(3)).sum())))
    result=dict(source_rows=len(frame),eligible_rows=len(data),excluded_rows=excluded,sessions=events,
        gate=bool(sum(r['passing_drivers']>=10 for r in events)>=2))
    return result,table,pd.concat(residuals,ignore_index=True)


def main():
    laps=pd.read_csv(ROOT/'results/e197_weather_aligned_laps.csv',float_precision='round_trip')
    weather=pd.read_csv(ROOT/'results/e197_weather_samples.csv',float_precision='round_trip')
    result,table,residuals=measure(attach(laps,weather))
    (ROOT/'results/e203_wind_support_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    table.to_csv(ROOT/'results/e203_wind_support.csv',index=False)
    residuals.to_csv(ROOT/'results/e203_wind_residuals.csv',index=False)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
