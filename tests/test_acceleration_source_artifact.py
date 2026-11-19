"""Independently rebuild E232 from per-lap raw arrays and speed interpolation."""
import json
from pathlib import Path
from unittest.mock import patch

import fastf1
import numpy as np
import pandas as pd

import f1.ff1
from experiments.e190_traffic_feasibility import cached_session


def test_source():
    saved=pd.read_csv('results/e232_acceleration_segments.csv',float_precision='round_trip')
    summary=json.loads(Path('results/e232_measurement_summary.json').read_text())
    assert not saved.duplicated(['round','driver','first_sample']).any()
    fastf1.Cache.offline_mode(True); total=0
    for round_ in range(1,22):
        session=cached_session(fastf1,2019,round_); raw=pd.DataFrame(session.laps)
        with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('audit HTTP forbidden')):
            cars=fastf1._api.car_data(session.api_path)
        origin=max((c.Date-c.Time).max() for c in cars.values() if c.Time.notna().any()).round('ms')
        keep=raw.LapNumber.gt(1)&raw.LapTime.notna()&raw.Time.notna()&raw.LapStartTime.notna()
        keep&=raw.PitInTime.isna()&raw.PitOutTime.isna()&raw.TrackStatus.astype(str).eq('1')&raw.IsAccurate.fillna(False)
        clean=raw[keep].copy(); clean['seconds']=clean.LapTime.dt.total_seconds()
        medians={driver:np.median(g.seconds) for driver,g in clean.groupby('Driver')}
        clean=clean[[r.seconds<1.07*medians[r.Driver] for r in clean.itertuples()]]
        rows=[]
        for driver,laps in clean.groupby('Driver',sort=True):
            car=cars.get(str(laps.DriverNumber.iloc[0]))
            if car is None:
                continue
            car=car[car.Date.notna()].drop_duplicates('Date',keep='last').sort_values('Date')
            all_t=(car.Date-origin).dt.total_seconds().to_numpy()
            all_ns=car.Date.to_numpy(dtype='datetime64[ns]').astype('int64')
            all_v=car.Speed.to_numpy(float)
            all_good=car.Time.notna().to_numpy()&car.Brake.eq(False).to_numpy()&car.Throttle.between(98,100).to_numpy()&np.isfinite(all_v)
            for lap in laps.itertuples():
                a,b=np.searchsorted(all_t,[lap.LapStartTime.total_seconds(),lap.Time.total_seconds()],side='left')
                t=all_t[a:b]; ns=all_ns[a:b]; v=all_v[a:b]; valid=all_good[a:b]
                for i in np.flatnonzero((v[:-1]<=100)&(v[1:]>100)):
                    lower=np.flatnonzero(v[i+1:]>=150)
                    if not len(lower):
                        continue
                    j=i+1+lower[0]
                    if j-i+1<4 or not valid[i:j+1].all():
                        continue
                    if not (np.diff(v[i:j+1])>=0).all() or not ((np.diff(ns[i:j+1])>0)&(np.diff(ns[i:j+1])<=500_000_000)).all():
                        continue
                    start=float(np.interp(100,[v[i],v[i+1]],[t[i],t[i+1]]))
                    end=float(np.interp(150,[v[j-1],v[j]],[t[j-1],t[j]]))
                    if .4<=end-start<=4:
                        rows.append(dict(first_sample=t[i],last_sample=t[j],start=start,end=end,duration=end-start,samples=j-i+1,driver=driver,lap=int(lap.LapNumber),season=2019,round=round_))
        actual=saved[saved['round'].eq(round_)]
        expected=pd.DataFrame(rows,columns=actual.columns)
        keys=['driver','lap','first_sample']
        pd.testing.assert_frame_equal(actual.sort_values(keys).reset_index(drop=True),expected[actual.columns].sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-10,rtol=1e-12)
        report=next(r for r in summary['source'] if r['round']==round_)
        assert report['segments']==len(expected) and report['driver_laps']==len(expected[['driver','lap']].drop_duplicates())
        total+=len(expected)
    assert total==summary['segments']==len(saved)


if __name__=='__main__':
    test_source()
    print('E232 independent all21 clean laps/raw-array acceleration crossings/clock/sample counts PASS')
