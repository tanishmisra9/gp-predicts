"""Independently reconstruct E235's fixed three raw launch windows."""
import json
from pathlib import Path
from unittest.mock import patch

import fastf1
import numpy as np
import pandas as pd

import f1.ff1
from experiments.e190_traffic_feasibility import cached_session


def independent(car,origin,t0):
    c=car.loc[car.Date.notna()].copy()
    c=c.loc[~c.Date.duplicated(keep='last')].sort_values('Date')
    ns=c.Date.to_numpy(dtype='datetime64[ns]').astype('int64')
    t=(ns-origin.value)/1e9
    v=c.Speed.to_numpy(float); seen=c.Time.notna().to_numpy()
    indices=[i for i in range(len(v)-1) if v[i]<=20<v[i+1] and t[i+1]>=t0 and t[i]<=t0+10]
    if not indices:
        return dict(supported=False,reason='no20crossing')
    i=indices[0]
    stops=[k for k in range(i+1) if v[k]<=2 and t0-2<=t[k]<=t0+2]
    if not stops:
        return dict(supported=False,reason='no_stationary')
    a=stops[-1]; ends=[k for k in range(i+1,len(v)) if v[k]>=40]
    if a==0 or v[a-1]>2 or t[a-1]<t0-2 or not ends:
        return dict(supported=False,reason='incomplete_launch')
    b=ends[0]
    okay=all(seen[k] and np.isfinite(v[k]) and v[k]>=0 for k in range(a-1,b+1))
    okay=okay and all(0<ns[k]-ns[k-1]<=500_000_000 for k in range(a,b+1))
    okay=okay and all(v[k]>=v[k-1] for k in range(a+1,b+1))
    if not okay:
        return dict(supported=False,reason='invalid_samples')
    x=float(np.interp(20,[v[i],v[i+1]],[t[i],t[i+1]]))
    y=float(np.interp(40,[v[b-1],v[b]],[t[b-1],t[b]]))
    if not t0<=x<y<=t0+10:
        return dict(supported=False,reason='outside_window')
    return dict(supported=True,reason='ok',first_sample=float(t[a-1]),last_sample=float(t[b]),time20=x,time40=y,samples=b-a+2)


def test_artifact():
    reports=json.loads(Path('results/e235_source_status.json').read_text())
    inventory=json.loads(Path('results/e212_source_status.json').read_text())
    assert len(reports)==103 and {(r['season'],r['round']) for r in reports}=={(r['season'],r['round']) for r in inventory}
    for report in reports:
        receipt=next(r for r in inventory if (r['season'],r['round'])==(report['season'],report['round']))
        if not receipt['available']:
            assert not report['available'] and report['reason']=='unavailable_in_frozen_cache_inventory' and not report['gate']
    result=dict(sessions=[r for r in reports if r['available']])
    assert all(2018<=r['season']<=2022 for r in result['sessions'])
    fastf1.Cache.offline_mode(True); passes=0
    for report in result['sessions']:
        session=cached_session(fastf1,report['season'],report['round'])
        with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('audit HTTP forbidden')):
            cars=fastf1._api.car_data(session.api_path)
        laps=pd.DataFrame(session.laps); first=laps.loc[laps.LapNumber.eq(1)]
        clock=[r.LapStartTime.total_seconds() for r in first.itertuples() if pd.isna(r.PitOutTime) and pd.notna(r.LapStartTime)]
        clock_ok=len(clock)>=12 and max(clock)-min(clock)<=.05
        t0=float(np.median(clock)) if clock else None
        origins=[(c.loc[c.Time.notna(),'Date']-c.loc[c.Time.notna(),'Time']).max() for c in cars.values() if c.Time.notna().any()]
        origin=max(origins).round('ms')
        assert report['clock_rows']==len(clock) and report['clock_ok']==clock_ok and report['t0']==t0
        assert len(first)==report['drivers']==len(report['rows'])
        assert set(first.DriverNumber.astype(str))=={r['number'] for r in report['rows']}
        expected=[]
        for raw in first.sort_values('DriverNumber').itertuples():
            number=str(raw.DriverNumber)
            if not clock_ok:
                row=dict(supported=False,reason='clock_unsupported')
            elif pd.notna(raw.PitOutTime):
                row=dict(supported=False,reason='pit_start')
            elif number not in cars:
                row=dict(supported=False,reason='missing_car')
            else:
                row=independent(cars[number],origin,t0)
            expected.append(dict(driver=str(raw.Driver),number=number,**row))
        good=[r for r in expected if r['supported']]
        spread=max(r['time20'] for r in good)-min(r['time20'] for r in good) if good else None
        center=float(np.median([r['time20'] for r in good])) if good else None
        for r in good:
            r['relative20']=r['time20']-center
        for actual,wanted in zip(report['rows'],expected):
            assert actual.keys()==wanted.keys()
            for key,value in wanted.items():
                if isinstance(value,float):
                    np.testing.assert_allclose(actual[key],value,atol=1e-10,rtol=1e-12)
                else:
                    assert actual[key]==value,(report['round'],key,actual[key],value)
        assert report['supported']==len(good)
        if good:
            np.testing.assert_allclose(report['spread20'],spread,atol=1e-10)
        else:
            assert report['spread20'] is None
        gate=bool(clock_ok and len(good)>=12 and spread<=2)
        assert report['gate']==gate
        passes+=gate
        print(report['round'],len(good),'supported, gate',gate)
    assert passes==sum(r['gate'] for r in reports)


if __name__=='__main__':
    test_artifact()
    print('E235 independent all-car clock/cohort/raw-sequence/centering/gate audit PASS')
