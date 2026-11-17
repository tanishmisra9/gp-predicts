"""E210 independently segment raw sample arrays for every pilot pit window."""
import json
from pathlib import Path
from unittest.mock import patch

import fastf1
import numpy as np
import pandas as pd

import f1.ff1
from experiments.e190_traffic_feasibility import cached_session
from experiments.e209_stationary_source import stationary


def test_artifacts():
    result=json.loads(Path('results/e210_measurement_summary.json').read_text())
    saved=pd.read_csv('results/e210_stationary_windows.csv',float_precision='round_trip')
    fastf1.Cache.offline_mode(True); matched_total=0; good_events=0; expected_windows=0
    reports=json.loads(Path('results/e210_source_status.json').read_text())
    for report in reports:
        season=report['season']; round_=report['round']
        if not report['available']:
            continue
        session=cached_session(fastf1,season,round_); laps=pd.DataFrame(session.laps)
        with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('audit HTTP forbidden')):
            cars=fastf1._api.car_data(session.api_path)
        offsets=[]
        for frame in cars.values():
            valid=frame.Time.notna()&frame.Date.notna()
            delta=frame.loc[valid,'Date'].to_numpy()-frame.loc[valid,'Time'].to_numpy()
            offsets.append(np.max(delta))
        origin=pd.Timestamp(max(offsets)).round('ms')
        assert report['available'] and origin==pd.Timestamp(report['origin'])
        assert report['car_rows']==sum(len(c) for c in cars.values())
        assert report['padded_rows']==sum(int(c.Time.isna().sum()) for c in cars.values())
        count=0; matched=0
        for driver,g in laps.groupby('Driver',sort=True):
            for pit in g[g.PitInTime.notna()].itertuples():
                exits=g[(g.LapNumber==pit.LapNumber+1)&g.PitOutTime.notna()]
                if len(exits)!=1 or '5' in str(pit.TrackStatus)+str(exits.TrackStatus.iloc[0]):
                    continue
                start=pit.PitInTime.total_seconds(); end=exits.PitOutTime.iloc[0].total_seconds()
                if not 10<end-start<60:
                    continue
                count+=1
                frame=cars[str(g.DriverNumber.iloc[0])]
                frame=frame[frame.Date.notna()].drop_duplicates('Date',keep='last').sort_values('Date')
                t=(frame.Date-origin).dt.total_seconds().to_numpy()
                select=(t>=start)&(t<=end); t=t[select]
                speed=frame.Speed.to_numpy()[select]; observed=frame.Time.notna().to_numpy()[select]
                runs=[]; first=None
                for i in range(len(t)+1):
                    zero=i<len(t) and observed[i] and speed[i]==0
                    broken=i>0 and i<len(t) and t[i]-t[i-1]>.5
                    if first is not None and (not zero or broken):
                        last=i-1
                        if first>0 and last<len(t)-1 and observed[first-1] and observed[last+1] and speed[first-1]>0 and speed[last+1]>0:
                            lower=t[last]-t[first]; upper=t[last+1]-t[first-1]
                            if t[first]-t[first-1]<=.5 and t[last+1]-t[last]<=.5 and 1<=lower<=10 and upper-lower<=.75:
                                runs.append(dict(first_zero=t[first],last_zero=t[last],left_moving=t[first-1],right_moving=t[last+1],lower=lower,upper=upper,midpoint=(lower+upper)/2,samples=last-first+1))
                        first=None
                    if zero and first is None:
                        first=i
                row=saved[saved['round'].eq(round_)&saved.driver.eq(driver)&saved.pitin_lap.eq(pit.LapNumber)]
                assert len(row)==1; row=row.iloc[0]
                assert row.window_rows==len(t) and row.padded_rows==int((~observed).sum()) and row.candidates==len(runs)
                assert row.usable==(len(runs)==1)
                np.testing.assert_allclose([row.pin,row.pout],[start,end],atol=1e-12)
                if len(runs)==1:
                    matched+=1
                    for name,value in runs[0].items():
                        np.testing.assert_allclose(row[name],value,atol=1e-10)
                    changed=frame.copy(); changed.loc[(changed.Date-origin).dt.total_seconds().gt(end),'Speed']=0
                    assert stationary(changed,start,end,origin)==stationary(frame,start,end,origin)
        assert report['pit_windows']==count and report['usable']==matched
        expected_windows+=count; matched_total+=matched
        good_events+=int(count>=15 and matched>=.7*count)
    assert expected_windows==len(saved)==result['pit_windows']
    assert matched_total==result['usable_windows']
    assert len(reports)==result['source_events']==21
    assert sum(r['available'] for r in reports)==result['available']
    pilot=pd.read_csv('results/e209_stationary_windows.csv',float_precision='round_trip')
    current=saved[saved['round'].isin([5,12,14])].copy()
    keys=['season','round','driver','pitin_lap']
    pd.testing.assert_frame_equal(current.sort_values(keys).reset_index(drop=True),pilot.sort_values(keys).reset_index(drop=True),check_dtype=False)


if __name__=='__main__':
    test_artifacts()
    print('E210 independent raw sample runs, clock/stop matching/brackets/coverage and unchanged pilot subset PASS')
