"""Independently rebuild E212 from per-lap raw arrays and speed interpolation."""
import json
from pathlib import Path
from unittest.mock import patch

import fastf1
import numpy as np
import pandas as pd

import f1.ff1
from experiments.e190_traffic_feasibility import cached_session


def test_source():
    saved=pd.read_csv('results/e212_braking_laps.csv',float_precision='round_trip')
    reports=json.loads(Path('results/e212_source_status.json').read_text())
    assert not saved.duplicated(['season','round','driver','lap']).any()
    fastf1.Cache.offline_mode(True); total=0
    for report in reports:
        season=report['season']; round_=report['round']
        if not report['available']:
            assert saved[saved.season.eq(season)&saved['round'].eq(round_)].empty
            continue
        session=cached_session(fastf1,season,round_); raw=pd.DataFrame(session.laps)
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
            all_good=car.Time.notna().to_numpy()&car.Brake.eq(True).to_numpy()&np.isfinite(all_v)
            for lap in laps.itertuples():
                a,b=np.searchsorted(all_t,[lap.LapStartTime.total_seconds(),lap.Time.total_seconds()],side='left')
                t=all_t[a:b]; ns=all_ns[a:b]; v=all_v[a:b]; valid=all_good[a:b]
                for i in np.flatnonzero((v[:-1]>=250)&(v[1:]<250)):
                    lower=np.flatnonzero(v[i+1:]<=150)
                    if not len(lower):
                        continue
                    j=i+1+lower[0]
                    if j-i+1<4 or not valid[i:j+1].all():
                        continue
                    if not (np.diff(v[i:j+1])<=0).all() or not ((np.diff(ns[i:j+1])>0)&(np.diff(ns[i:j+1])<=500_000_000)).all():
                        continue
                    start=float(np.interp(250,[v[i+1],v[i]],[t[i+1],t[i]]))
                    end=float(np.interp(150,[v[j],v[j-1]],[t[j],t[j-1]]))
                    if .4<=end-start<=3:
                        rows.append(dict(first_sample=t[i],last_sample=t[j],start=start,end=end,duration=end-start,samples=j-i+1,driver=driver,lap=int(lap.LapNumber),season=season,round=round_))
        expected=pd.DataFrame(rows,columns=['driver','lap','duration'])
        segment_count=len(expected)
        expected=expected.groupby(['driver','lap'],sort=True).duration.median().reset_index().assign(season=season,round=round_)
        actual=saved[saved.season.eq(season)&saved['round'].eq(round_)]
        keys=['driver','lap']
        pd.testing.assert_frame_equal(actual.sort_values(keys).reset_index(drop=True),expected[actual.columns].sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-10,rtol=1e-12)
        assert report['segments']==segment_count and report['driver_laps']==len(expected)
        assert report['car_rows']==sum(len(c) for c in cars.values())
        assert report['padded_rows']==sum(int(c.Time.isna().sum()) for c in cars.values())
        total+=len(expected)
    assert total==len(saved)
    old=pd.read_csv('results/e211_braking_segments.csv',float_precision='round_trip')
    old=old.groupby(['driver','lap','season','round'],sort=True).duration.median().reset_index()
    current=saved[saved.season.eq(2019)]
    keys=['season','round','driver','lap']
    pd.testing.assert_frame_equal(current.sort_values(keys).reset_index(drop=True),old[current.columns].sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=1e-12)


if __name__=='__main__':
    test_source()
    print('E212 independent all available raw laps/crossings/medians/counts and unchanged 2019 source PASS')
