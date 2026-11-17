"""E209 fixed historical pit stationary-interval pilot; no winner model."""
import json
import os
from unittest.mock import patch

import fastf1
import pandas as pd
import requests

from experiments.e190_traffic_feasibility import PILOT,cached_session
from experiments.e196_weather_source import restricted_send
from f1.data import ROOT


def allowed_urls(path):
    assert path.startswith('/static/2019/') and path.endswith('/')
    return {host+path+'CarData.z.jsonStream' for host in ('https://livetiming.formula1.com','https://livetiming-mirror.fastf1.dev')}


def stationary(car,pin,pout,origin):
    data=car[car.Date.notna()].drop_duplicates('Date',keep='last').sort_values('Date').copy()
    data['s']=(data.Date-origin).dt.total_seconds()
    data=data[data.s.between(pin,pout)].reset_index(drop=True)
    zero=data.Time.notna()&data.Speed.eq(0)
    group=(~zero|~zero.shift(1,fill_value=False)|data.s.diff().gt(.5)).cumsum()
    runs=[]
    for _,g in data[zero].groupby(group[zero],sort=True):
        first=int(g.index[0]); last=int(g.index[-1])
        if first==0 or last==len(data)-1:
            continue
        left=data.iloc[first-1]; right=data.iloc[last+1]
        moving=pd.notna(left.Time) and pd.notna(right.Time) and left.Speed>0 and right.Speed>0
        if not moving or g.s.iloc[0]-left.s>.5 or right.s-g.s.iloc[-1]>.5:
            continue
        lower=float(g.s.iloc[-1]-g.s.iloc[0]); upper=float(right.s-left.s)
        if 1<=lower<=10 and upper-lower<=.75:
            runs.append(dict(first_zero=float(g.s.iloc[0]),last_zero=float(g.s.iloc[-1]),
                left_moving=float(left.s),right_moving=float(right.s),lower=lower,upper=upper,
                midpoint=(lower+upper)/2,samples=len(g)))
    return dict(window_rows=len(data),padded_rows=int(data.Time.isna().sum()),candidates=len(runs),
        usable=len(runs)==1,**(runs[0] if len(runs)==1 else {}))


def measure(laps,cars):
    offsets=[(car.Date-car.Time).max() for car in cars.values() if car.Time.notna().any()]
    assert offsets and all(pd.notna(x) for x in offsets)
    origin=max(offsets).round('ms'); rows=[]
    for driver,g in laps.groupby('Driver',sort=True):
        assert g.DriverNumber.astype(str).nunique()==1
        number=str(g.DriverNumber.iloc[0]); car=cars.get(number)
        for row in g[g.PitInTime.notna()].itertuples():
            after=g[g.LapNumber.eq(row.LapNumber+1)&g.PitOutTime.notna()]
            if len(after)!=1 or '5' in str(row.TrackStatus)+str(after.TrackStatus.iloc[0]):
                continue
            pin=row.PitInTime.total_seconds(); pout=after.PitOutTime.iloc[0].total_seconds()
            if not 10<pout-pin<60:
                continue
            values=stationary(car,pin,pout,origin) if car is not None else dict(window_rows=0,padded_rows=0,candidates=0,usable=False)
            rows.append(dict(driver=driver,pitin_lap=int(row.LapNumber),pin=pin,pout=pout,**values))
    return pd.DataFrame(rows),origin


def main():
    import f1.ff1
    from fastf1.internals.f1account import get_token
    if not all(os.environ.get(k) for k in ('F1_USERNAME','F1_PASSWORD')):
        raise RuntimeError('Missing existing F1 credentials')
    fastf1.Cache.offline_mode(False)
    try:
        get_token(method='password')
    except Exception as exc:
        result=dict(sessions=[],gate=False,error_type=type(exc).__name__)
        (ROOT/'results/e209_stationary_summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True); return
    reports=[]; parts=[]
    for season,round_ in PILOT:
        fastf1.Cache.offline_mode(True)
        session=cached_session(fastf1,season,round_); laps=pd.DataFrame(session.laps)
        report=dict(season=season,round=round_,available=False,requests=[])
        fastf1.Cache.offline_mode(False)
        try:
            with patch('requests.adapters.HTTPAdapter.send',new=restricted_send(requests.adapters.HTTPAdapter.send,allowed_urls(session.api_path),report['requests'])):
                cars=fastf1._api.car_data(session.api_path)
        except Exception as exc:
            report['error_type']=type(exc).__name__
        else:
            measured,origin=measure(laps,cars); parts.append(measured.assign(season=season,round=round_))
            report.update(available=True,car_rows=sum(len(c) for c in cars.values()),
                padded_rows=sum(int(c.Time.isna().sum()) for c in cars.values()),
                pit_windows=len(measured),usable=int(measured.usable.sum()),origin=str(origin))
        reports.append(report); print(json.dumps(report),flush=True)
        (ROOT/'results/e209_stationary_progress.json').write_text(json.dumps(reports,indent=2)+'\n')
    table=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    good=sum(r.get('pit_windows',0)>=15 and r.get('usable',0)>=.7*r.get('pit_windows',1) for r in reports)
    result=dict(sessions=reports,gate=bool(good>=2 and sum(r.get('usable',0) for r in reports)>=30))
    table.to_csv(ROOT/'results/e209_stationary_windows.csv',index=False)
    (ROOT/'results/e209_stationary_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
