"""E198: fixed thermal design support only; never fit a lap-time response."""
import json

import numpy as np
import pandas as pd

from experiments.e190_traffic_feasibility import PILOT
from f1.data import ROOT


def prepare(frame):
    assert set(map(tuple,frame[['season','round']].drop_duplicates().to_numpy()))==set(PILOT)
    assert not frame.duplicated(['season','round','Driver','LapNumber']).any()
    out=frame.copy()
    for name in ('LapStartTime','weather_time'):
        out[name]=pd.to_timedelta(out[name])
    finite=np.isfinite(out[['TrackTemp','TyreLife','Position']]).all(axis=1)
    valid=finite&out.LapStartTime.notna()&out.weather_time.notna()&out.Stint.notna()&out.Compound.notna()&out.close.isin([True,False])&out.Rainfall.eq(False)
    out=out[valid].copy()
    out['phase']=0.; out['half']=0
    for _,g in out.groupby(['season','round'],sort=True):
        start=g.LapStartTime.dt.total_seconds()
        span=start.max()-start.min()
        out.loc[g.index,'phase']=(start-start.min())/span if span>0 else 0.
        weather=sorted(g.weather_time.unique())
        split={time:i%2 for i,time in enumerate(weather)}
        out.loc[g.index,'half']=g.weather_time.map(split)
    return out,int((~valid).sum())


def design(g):
    phase=g.phase.to_numpy(float); age=g.TyreLife.to_numpy(float)/10
    columns=[np.ones(len(g)),phase,phase**2,phase**3,g.Position.to_numpy(float)/20,g.close.to_numpy(float)]
    stints=sorted(g.Stint.unique())
    columns.extend(g.Stint.eq(stint).to_numpy(float) for stint in stints[1:])
    for compound in sorted(g.Compound.unique()):
        mask=g.Compound.eq(compound).to_numpy(float)
        columns.extend([age*mask,age**2*mask])
    return np.column_stack(columns)


def support(g):
    Z=design(g); temperature=g.TrackTemp.to_numpy(float)
    if len(g):
        beta,_,rank,_=np.linalg.lstsq(Z,temperature,rcond=1e-10)
        residual=temperature-Z@beta
        rms=float(np.sqrt(np.mean(residual**2)))
        variance=float(np.var(temperature))
        fraction=float(np.mean(residual**2)/variance) if variance>1e-12 else 0.
    else:
        residual=np.array([]); rank=0; rms=np.nan; fraction=0.
    n=len(g); readings=g.weather_time.nunique(); dof=n-int(rank)
    passed=bool(n>=15 and readings>=10 and dof>=10 and rms>=.5 and fraction>=.10)
    return dict(rows=n,readings=int(readings),rank=int(rank),dof=dof,residual_rms=rms,variance_fraction=fraction,support=passed),residual


def measure(frame):
    data,excluded=prepare(frame); reports=[]; residuals=[]
    for season,round_,driver in frame[['season','round','Driver']].drop_duplicates().itertuples(index=False,name=None):
        g=data[data.season.eq(season)&data['round'].eq(round_)&data.Driver.eq(driver)]
        for name,subset in [('full',g),('half0',g[g.half.eq(0)]),('half1',g[g.half.eq(1)])]:
            values,residual=support(subset)
            reports.append(dict(season=int(season),round=int(round_),driver=driver,subset=name,**values))
            rows=subset[['season','round','Driver','LapNumber','weather_time','TrackTemp','phase','half']].copy()
            rows['subset']=name; rows['residual_temperature']=residual
            residuals.append(rows)
    table=pd.DataFrame(reports)
    events=[]
    for season,round_ in PILOT:
        event=table[table.season.eq(season)&table['round'].eq(round_)]
        passed=event.groupby('driver').support.agg(['all','size'])
        count=int((passed['all']&passed['size'].eq(3)).sum())
        events.append(dict(season=season,round=round_,drivers=int(len(passed)),passing_drivers=count))
    result=dict(source_rows=len(frame),eligible_rows=len(data),excluded_rows=excluded,sessions=events,gate=bool(sum(r['passing_drivers']>=10 for r in events)>=2))
    return result,table,pd.concat(residuals,ignore_index=True)


def main():
    source=pd.read_csv(ROOT/'results/e197_weather_aligned_laps.csv',float_precision='round_trip')
    result,table,residuals=measure(source)
    (ROOT/'results/e198_temperature_support_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    table.to_csv(ROOT/'results/e198_temperature_support.csv',index=False)
    residuals.to_csv(ROOT/'results/e198_temperature_residuals.csv',index=False)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
