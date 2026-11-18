"""Independently reconstruct E213 coast runs backwards from brake onset."""
import json
from pathlib import Path
from unittest.mock import patch

import fastf1
import numpy as np
import pandas as pd

import f1.ff1
from experiments.e190_traffic_feasibility import cached_session


def test_source():
    saved=pd.read_csv('results/e213_coasting_laps.csv',float_precision='round_trip')
    assert not saved.duplicated(['season','round','driver','lap']).any()
    result=json.loads(Path('results/e213_measurement_summary.json').read_text())
    fastf1.Cache.offline_mode(True); total_runs=0; total_laps=0
    for round_ in range(1,22):
        session=cached_session(fastf1,2019,round_); laps=pd.DataFrame(session.laps)
        with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('coast audit HTTP forbidden')):
            cars=fastf1._api.car_data(session.api_path)
        origin=max((c.Date-c.Time).max() for c in cars.values() if c.Time.notna().any()).round('ms')
        valid=laps.LapNumber.gt(1)&laps.LapTime.notna()&laps.Time.notna()&laps.LapStartTime.notna()
        valid&=laps.PitInTime.isna()&laps.PitOutTime.isna()&laps.TrackStatus.astype(str).eq('1')&laps.IsAccurate.fillna(False)
        clean=laps[valid].copy(); clean['seconds']=clean.LapTime.dt.total_seconds()
        medians={d:np.median(g.seconds) for d,g in clean.groupby('Driver')}
        rows=[]
        for driver,g in clean.groupby('Driver',sort=True):
            c=cars.get(str(g.DriverNumber.iloc[0]))
            if c is None:
                continue
            c=c[c.Date.notna()].drop_duplicates('Date',keep='last').sort_values('Date')
            times=(c.Date-origin).dt.total_seconds().to_numpy()
            nanos=c.Date.to_numpy(dtype='datetime64[ns]').astype('int64')
            speed=c.Speed.to_numpy(float); throttle=c.Throttle.to_numpy(float)
            brake=c.Brake.to_numpy(); observed=c.Time.notna().to_numpy()
            for lap in g.itertuples():
                if lap.seconds>=1.07*medians[driver]:
                    continue
                start=lap.LapStartTime.total_seconds(); end=lap.Time.total_seconds()
                a,b=np.searchsorted(times,[start,end],side='left')
                t=times[a:b]; ns=nanos[a:b]; v=speed[a:b]; th=throttle[a:b]; br=brake[a:b]
                good=observed[a:b]&np.isfinite(v)&np.isfinite(th)&(v>=0)&(th>=0)&(th<=100)&pd.Series(br).isin([True,False]).to_numpy()
                usable=[i for i in range(len(t)-1) if good[i] and good[i+1] and 0<ns[i+1]-ns[i]<=500_000_000]
                coverage=sum((ns[i+1]-ns[i])/1e9 for i in usable)/(end-start)
                if coverage<.9:
                    continue
                coast=good&(v>=200)&(th<=5)&(br==False)
                seconds=0.; runs=0
                for j in np.flatnonzero(good&(v>=200)&(br==True)):
                    i=j-1
                    while i>=0 and coast[i]:
                        i-=1
                    first=i+1
                    if i<0 or j-first<3 or not (good[i] and th[i]>5 and br[i]==False and v[i]>=200):
                        continue
                    if not all(k in usable for k in range(i,j)):
                        continue
                    duration=t[j]-t[first]
                    if .4<=duration<=3:
                        seconds+=duration; runs+=1
                rows.append(dict(driver=driver,lap=int(lap.LapNumber),first_sample=start,
                    duration=100*seconds/(end-start),coast_seconds=seconds,coverage=coverage,runs=runs,
                    season=2019,round=round_))
        expected=pd.DataFrame(rows)
        actual=saved[saved['round'].eq(round_)]
        keys=['driver','lap']
        pd.testing.assert_frame_equal(actual.sort_values(keys).reset_index(drop=True),expected[actual.columns].sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=1e-12)
        report=result['source'][round_-1]
        assert report==dict(round=round_,laps=len(expected),runs=int(expected.runs.sum()))
        total_runs+=int(expected.runs.sum()); total_laps+=len(expected)
    assert result['runs']==total_runs and result['laps']==total_laps==len(saved)


if __name__=='__main__':
    test_source()
    print('E213 independent all21raw backwards coast runs, observed coverage, zeros and lap counts PASS')
