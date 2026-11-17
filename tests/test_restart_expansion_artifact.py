"""Independent checkpoint binary searches and cohort ranks for E207."""
import json
from pathlib import Path
from unittest.mock import patch

import fastf1
import numpy as np
import pandas as pd

import f1.ff1
from experiments.e190_traffic_feasibility import cached_session
from experiments.e206_restart_source import COLS,extract_restarts


def test_artifacts():
    result=json.loads(Path('results/e207_source_summary.json').read_text())
    saved=pd.read_csv('results/e207_restart_source.csv',float_precision='round_trip')
    for col in ('green','before','after'):
        saved[col]=pd.to_timedelta(saved[col])
    expected=[]; reports=[]; fastf1.Cache.offline_mode(True)
    for report in result['sessions']:
        season=report['season']; round_=report['round']
        if not report['available']:
            try:
                session=cached_session(fastf1,season,round_); raw_laps=pd.DataFrame(session.laps)
                raw_flags=pd.DataFrame(session.track_status)
                assert raw_flags.Time.notna().all() and not raw_flags.Time.duplicated().any()
                with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('audit HTTP forbidden')):
                    fastf1._api._extended_timing_data(session.api_path)
            except Exception:
                continue
            raise AssertionError('recorded unavailable source is independently available')
        session=cached_session(fastf1,season,round_); laps=pd.DataFrame(session.laps)
        flags=session.track_status.sort_values('Time'); ts=flags.Time.dt.total_seconds().to_numpy(); codes=flags.Status.to_numpy()
        with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('audit HTTP forbidden')):
            _,raw,_=fastf1._api._extended_timing_data(session.api_path)
        identities=laps[['DriverNumber','Driver']].drop_duplicates()
        mapping=dict(zip(identities.DriverNumber.astype(str),identities.Driver))
        stream=raw.copy(); stream.Driver=stream.Driver.astype(str).map(mapping)
        stream=stream.drop_duplicates(['Driver','Time'],keep='last')
        groups={d:g.sort_values('Time') for d,g in stream.groupby('Driver')}
        events=[]; selected=0
        for index in np.flatnonzero((codes=='1')&np.r_[False,codes[:-1]=='4']):
            number=len(events)+1; t=ts[index]; before=t-5; after=t+120
            prior=np.searchsorted(ts,before,side='right')-1
            clear=bool(prior>=0 and codes[prior]=='4' and np.all(codes[(ts>t)&(ts<=after)]=='1'))
            candidates=[]
            if clear:
                for driver,g in groups.items():
                    times=g.Time.dt.total_seconds().to_numpy()
                    indices=np.searchsorted(times,[before,after],side='right')-1
                    if (indices<0).any():
                        continue
                    ages=np.array([before,after])-times[indices]
                    positions=g.Position.to_numpy()[indices]
                    if not ((ages>=0)&(ages<=30)).all() or not np.isfinite(positions).all() or not ((positions>=1)&(positions<=30)&(positions==np.floor(positions))).all():
                        continue
                    own=laps[laps.Driver.eq(driver)]; end=own.Time.dt.total_seconds(); start=own.LapStartTime.dt.total_seconds()
                    overlap=own[(end>=before)&(start<=after)]
                    full=(start>=t)&(end<=after)&own.TrackStatus.astype(str).eq('1')&own.IsAccurate.fillna(False)
                    if overlap.empty or overlap.PitInTime.notna().any() or overlap.PitOutTime.notna().any() or not full.any() or end.max()<after:
                        continue
                    candidates.append(dict(driver=driver,position_before=positions[0],position_after=positions[1],age_before=ages[0],age_after=ages[1]))
            unique=bool(candidates and len({r['position_before'] for r in candidates})==len(candidates) and len({r['position_after'] for r in candidates})==len(candidates))
            usable=bool(clear and unique and len(candidates)>=6)
            events.append(dict(restart=number,green_seconds=t,clear_window=clear,eligible_drivers=len(candidates),unique_positions=unique,usable=usable))
            if usable:
                selected+=len(candidates)
                for row in candidates:
                    row['rank_before']=1+sum(other['position_before']<row['position_before'] for other in candidates)
                    row['rank_after']=1+sum(other['position_after']<row['position_after'] for other in candidates)
                    row['gain']=row['rank_before']-row['rank_after']
                    row.update(restart=number,green=pd.Timedelta(seconds=t),before=pd.Timedelta(seconds=before),after=pd.Timedelta(seconds=after),season=season,round=round_)
                    expected.append(row)
        reports.append(dict(season=season,round=round_,events=events,rows=selected))
        assert events==report['events'] and selected==report['rows']
    expected=pd.DataFrame(expected)[COLS+['season','round']].sort_values(['season','round','restart','driver']).reset_index(drop=True)
    actual=saved.sort_values(['season','round','restart','driver']).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False,atol=1e-8,rtol=1e-10)
    assert len(actual)==result['opportunities']
    counts=[sum(e['usable'] for e in r['events']) for r in reports]
    assert result['available']==len(reports)
    assert result['usable_restarts']==sum(counts)
    assert result['races']==sum(c>0 for c in counts)
    assert result['coverage_gate']==bool(len(reports)>=95 and sum(counts)>=20 and sum(c>0 for c in counts)>=15 and len(actual)>=300)
    assert actual.groupby(['season','round','restart']).gain.sum().eq(0).all()
    pilot=pd.read_csv('results/e206_restart_source.csv',float_precision='round_trip')
    subset=saved[saved.season.eq(2019)&saved['round'].isin([5,10,20])].copy()
    for col in ('green','before','after'):
        pilot[col]=pd.to_timedelta(pilot[col])
    keys=['season','round','restart','driver']
    pd.testing.assert_frame_equal(subset.sort_values(keys).reset_index(drop=True),pilot.sort_values(keys).reset_index(drop=True),check_dtype=False)


if __name__=='__main__':
    test_artifacts()
    print('E207 all available raw status/checkpoint/cohort/rank counts and source coverage gate PASS')
