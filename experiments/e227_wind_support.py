"""Design-only wind-speed-squared identifiability; fixed cached 2019 pilot, no pace fit."""
import json
import numpy as np
import pandas as pd
from experiments.e190_traffic_feasibility import PILOT
from experiments.e198_temperature_support import prepare,design
from f1.data import ROOT


def attach(laps,weather):
    source=weather[['season','round','Time','AirTemp','WindDirection','WindSpeed']].copy()
    source['weather_time']=pd.to_timedelta(source.pop('Time'))
    assert not source.duplicated(['season','round','weather_time']).any()
    out=laps.copy();out['weather_time']=pd.to_timedelta(out.weather_time)
    return out.merge(source,on=['season','round','weather_time'],how='left',validate='many_to_one')


def eligible(frame):
    base,excluded=prepare(frame)
    lag=(base.LapStartTime-base.weather_time).dt.total_seconds()
    keep=np.isfinite(base[['AirTemp','WindSpeed','WindDirection']]).all(axis=1)
    keep&=base.WindSpeed.between(0,40)&base.WindDirection.between(0,359)&lag.between(0,120)
    out=base[keep].copy();out['wind_load']=out.WindSpeed**2
    return out,excluded,int((~keep).sum())


def nuisance(g):
    theta=np.deg2rad(g.WindDirection.to_numpy(float))
    return np.column_stack([design(g),g.TrackTemp.to_numpy(float)/50,g.AirTemp.to_numpy(float)/30,
                            np.sin(theta),np.cos(theta)])


def support(g):
    n=len(g);Z=nuisance(g);load=g.wind_load.to_numpy(float)
    if n:
        beta,_,rank,_=np.linalg.lstsq(Z,load,rcond=1e-10)
        residual=load-Z@beta
        energy=float(np.mean(residual**2));var=float(np.var(load))
        rms=float(np.sqrt(energy));fraction=energy/var if var>1e-12 else 0.
    else:residual=np.array([]);rank=0;rms=None;fraction=0.
    readings=int(g.weather_time.nunique());dof=n-int(rank)
    passed=bool(n>=15 and readings>=10 and dof>=10 and rms>=1. and fraction>=.10)
    return dict(rows=n,readings=readings,rank=int(rank),dof=dof,residual_rms=rms,
                variance_fraction=fraction,support=passed),residual


def measure(frame):
    data,base_excluded,wind_excluded=eligible(frame);reports=[];parts=[]
    for season,round_,driver in frame[['season','round','Driver']].drop_duplicates().itertuples(index=False,name=None):
        g=data[data.season.eq(season)&data['round'].eq(round_)&data.Driver.eq(driver)]
        for name,subset in [('full',g),('half0',g[g.half.eq(0)]),('half1',g[g.half.eq(1)])]:
            stats,residual=support(subset)
            reports.append(dict(season=int(season),round=int(round_),driver=driver,subset=name,**stats))
            part=subset[['season','round','Driver','LapNumber','weather_time','phase','half','WindSpeed','WindDirection','AirTemp','TrackTemp','wind_load']].copy()
            part['subset']=name;part['residual_wind_load']=residual;parts.append(part)
    table=pd.DataFrame(reports);events=[]
    for season,round_ in PILOT:
        event=table[table.season.eq(season)&table['round'].eq(round_)]
        counts=event.groupby('driver').support.agg(['all','size'])
        events.append(dict(season=season,round=round_,drivers=len(counts),
                           passing_drivers=int((counts['all']&counts['size'].eq(3)).sum())))
    result=dict(source_rows=len(frame),eligible_rows=len(data),base_excluded=base_excluded,wind_excluded=wind_excluded,
                sessions=events,gate=bool(sum(r['passing_drivers']>=10 for r in events)>=2),pace_response_fit=False,winner_fit=False)
    return result,table,pd.concat(parts,ignore_index=True)


def main():
    laps=pd.read_csv(ROOT/'results/e197_weather_aligned_laps.csv',float_precision='round_trip')
    weather=pd.read_csv(ROOT/'results/e197_weather_samples.csv',float_precision='round_trip')
    frame=attach(laps,weather);result,table,residual=measure(frame)
    table.to_csv(ROOT/'results/e227_wind_support.csv',index=False)
    residual.to_csv(ROOT/'results/e227_wind_residuals.csv',index=False)
    (ROOT/'results/e227_wind_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
