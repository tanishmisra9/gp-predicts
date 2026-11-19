"""E233 stationary launch timing observability, three fixed cached 2019 races."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.e190_traffic_feasibility import PILOT,cached_session
from f1.data import ROOT


def launch(car,origin,t0):
    c=car[car.Date.notna()].drop_duplicates('Date',keep='last').sort_values('Date')
    t=(c.Date-origin).dt.total_seconds().to_numpy()
    ns=c.Date.to_numpy(dtype='datetime64[ns]').astype('int64')
    v=c.Speed.to_numpy(float); observed=c.Time.notna().to_numpy()&np.isfinite(v)&(v>=0)
    candidates=np.flatnonzero((v[:-1]<=20)&(v[1:]>20)&(t[1:]>=t0)&(t[:-1]<=t0+10))
    if not len(candidates):
        return dict(supported=False,reason='no20crossing')
    i=int(candidates[0])
    stationary=np.flatnonzero((v[:i+1]<=2)&(t[:i+1]>=t0-2)&(t[:i+1]<=t0+2))
    if not len(stationary):
        return dict(supported=False,reason='no_stationary')
    a=int(stationary[-1]); high=np.flatnonzero(v[i+1:]>=40)
    if a<1 or v[a-1]>2 or t[a-1]<t0-2 or not len(high):
        return dict(supported=False,reason='incomplete_launch')
    b=int(i+1+high[0]); gaps=np.diff(ns[a-1:b+1])
    if not observed[a-1:b+1].all() or not ((gaps>0)&(gaps<=500_000_000)).all() or not (np.diff(v[a:b+1])>=0).all():
        return dict(supported=False,reason='invalid_samples')
    time20=t[i]+(20-v[i])/(v[i+1]-v[i])*(t[i+1]-t[i])
    time40=t[b-1]+(40-v[b-1])/(v[b]-v[b-1])*(t[b]-t[b-1])
    if not t0<=time20<time40<=t0+10:
        return dict(supported=False,reason='outside_window')
    return dict(supported=True,reason='ok',first_sample=float(t[a-1]),last_sample=float(t[b]),
        time20=float(time20),time40=float(time40),samples=b-a+2)


def event(laps,cars):
    first=laps[laps.LapNumber.eq(1)].copy()
    assert not first.DriverNumber.duplicated().any()
    clock=first[first.PitOutTime.isna()].LapStartTime.dropna().dt.total_seconds()
    origin=max((c.Date-c.Time).max() for c in cars.values() if c.Time.notna().any()).round('ms')
    clock_ok=bool(len(clock)>=12 and clock.max()-clock.min()<=.05)
    t0=float(clock.median()) if len(clock) else None
    rows=[]
    for r in first.sort_values('DriverNumber').itertuples():
        number=str(r.DriverNumber)
        if not clock_ok:
            item=dict(supported=False,reason='clock_unsupported')
        elif pd.notna(r.PitOutTime):
            item=dict(supported=False,reason='pit_start')
        elif number not in cars:
            item=dict(supported=False,reason='missing_car')
        else:
            item=launch(cars[number],origin,t0)
        rows.append(dict(driver=str(r.Driver),number=number,**item))
    good=[r for r in rows if r['supported']]
    spread=float(np.ptp([r['time20'] for r in good])) if good else None
    gate=bool(clock_ok and len(good)>=12 and spread<=2)
    center=float(np.median([r['time20'] for r in good])) if good else None
    for r in good:
        r['relative20']=r['time20']-center
    return dict(clock_rows=len(clock),clock_ok=clock_ok,t0=t0,drivers=len(first),supported=len(good),spread20=spread,gate=gate,rows=rows)


def main():
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True); reports=[]
    for season,round_ in PILOT:
        session=cached_session(fastf1,season,round_)
        with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('launch HTTP forbidden')):
            cars=fastf1._api.car_data(session.api_path)
        report=event(pd.DataFrame(session.laps),cars)
        reports.append(dict(season=season,round=round_,**report))
        print(season,round_,report['supported'],report['gate'],flush=True)
    result=dict(sessions=reports,gate=sum(r['gate'] for r in reports)>=2,winner_fit=False,reaction_time_verified=False)
    (ROOT/'results/e233_launch_source.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
