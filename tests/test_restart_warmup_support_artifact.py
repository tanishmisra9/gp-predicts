"""Independent cached E237 episode/window and paired-team support audit."""
import json
from pathlib import Path
from collections import defaultdict

import fastf1
import numpy as np
import pandas as pd

import f1.ff1
from experiments.e190_traffic_feasibility import cached_session
from f1.data import load_results


def test_artifacts():
    result=json.loads(Path('results/e237_support_summary.json').read_text())
    saved=pd.read_csv('results/e237_restart_windows.csv',float_precision='round_trip')
    actual_pairs=pd.read_csv('results/e237_paired_teams.csv')
    entries=load_results().loc[lambda d:d.season.between(2018,2022),['season','round','code','team']]
    assert len(result['sessions'])==103 and {(r['season'],r['round']) for r in result['sessions']}==set(map(tuple,entries[['season','round']].to_numpy()))
    fastf1.Cache.offline_mode(True); expected=[]
    for report in result['sessions']:
        season,round_=report['season'],report['round']
        if not report['available']:
            assert saved[saved.season.eq(season)&saved['round'].eq(round_)].empty
            continue
        session=cached_session(fastf1,season,round_)
        laps=pd.DataFrame(session.laps); flags=pd.DataFrame(session.track_status).sort_values('Time',kind='stable')
        changes=[]
        for row in flags.itertuples():
            if not changes or row.Status!=changes[-1][1]:
                changes.append((row.Time,row.Status))
        episodes=[]
        for before,after in zip(changes,changes[1:]):
            duration=(after[0]-before[0]).total_seconds()
            if before[1]=='4' and after[1]=='1' and duration>=60:
                episodes.append((after[0],duration))
        mask=(laps.LapNumber>1)&laps.LapTime.notna()&laps.Time.notna()&laps.LapStartTime.notna()
        mask&=laps.PitInTime.isna()&laps.PitOutTime.isna()&laps.TrackStatus.astype(str).eq('1')&laps.IsAccurate.fillna(False)
        basic=laps[mask]
        medians={driver:np.median(g.LapTime.dt.total_seconds()) for driver,g in basic.groupby('Driver')}
        clean={(r.Driver,int(r.LapNumber)) for r in basic.itertuples() if r.LapTime.total_seconds()<1.07*medians[r.Driver]}
        rows=[]; episode_reports=[]
        for number,(green,duration) in enumerate(episodes,1):
            count=0
            for driver,g in laps.groupby('Driver',sort=True):
                by_lap={int(r.LapNumber):r for r in g.itertuples()}
                candidates=[n for n,r in by_lap.items() if pd.notna(r.LapStartTime) and r.LapStartTime>=green]
                if not candidates: continue
                n=min(candidates)
                if n-1 not in by_lap or not all((driver,n+i) in clean for i in range(4)): continue
                prior=by_lap[n-1]; four=[by_lap[n+i] for i in range(4)]
                if pd.isna(prior.LapStartTime) or not prior.LapStartTime<=green<=prior.Time or '4' not in str(prior.TrackStatus): continue
                if four[0].LapStartTime>green+pd.Timedelta(seconds=200): continue
                if pd.notna(prior.PitInTime) or pd.notna(prior.PitOutTime): continue
                if prior.Compound not in ('SOFT','MEDIUM','HARD') or any(r.Compound!=prior.Compound or r.Stint!=prior.Stint for r in four): continue
                ages=np.asarray([prior.TyreLife]+[r.TyreLife for r in four],dtype=float)
                if not np.isfinite(ages).all() or not np.allclose(np.diff(ages),1): continue
                positions=[r.Position for r in four]
                if any(pd.isna(p) for p in positions) or len(set(positions))!=1: continue
                if any(r.Status!='1' for r in flags.itertuples() if green<r.Time<=four[-1].Time): continue
                rows.append(dict(restart=number,green_seconds=green.total_seconds(),sc_seconds=duration,driver=driver,
                    previous_lap=n-1,first_full_lap=n,stint=prior.Stint,compound=prior.Compound,season=season,round=round_))
                count+=1
            episode_reports.append(dict(restart=number,green_seconds=green.total_seconds(),sc_seconds=duration,eligible_drivers=count))
        assert report['restarts']==episode_reports and report['blocks']==len(rows)
        expected.extend(rows)
    rebuilt=pd.DataFrame(expected,columns=saved.columns)
    keys=['season','round','restart','driver']
    pd.testing.assert_frame_equal(saved.sort_values(keys).reset_index(drop=True),rebuilt.sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-10,rtol=1e-12)
    mapping={(r.season,r.round,r.code):r.team for r in entries.itertuples()}
    rosters=defaultdict(set); cohorts=defaultdict(set)
    for r in entries.itertuples(): rosters[(r.season,r.round,r.team)].add(r.code)
    for r in rebuilt.itertuples():
        team=mapping[(r.season,r.round,r.driver)]
        cohorts[(r.season,r.round,r.restart,team)].add(r.driver)
    pairs=[dict(season=s,round=r,restart=e,team=t) for (s,r,e,t),drivers in sorted(cohorts.items()) if len(rosters[(s,r,t)])==2 and drivers==rosters[(s,r,t)]]
    pair_frame=pd.DataFrame(pairs,columns=actual_pairs.columns)
    pd.testing.assert_frame_equal(actual_pairs,pair_frame,check_dtype=False)
    episode_teams=defaultdict(set)
    for p in pairs: episode_teams[(p['season'],p['round'],p['restart'])].add(p['team'])
    qualified={event:teams for event,teams in episode_teams.items() if len(teams)>=6}
    race_keys={(s,r) for s,r,e in qualified}
    team_races={(s,r,t) for (s,r,e),teams in qualified.items() for t in teams}
    assert result['blocks']==len(saved) and result['paired_team_restarts']==len(pairs)
    assert result['qualified_restarts']==len(qualified) and result['qualified_races']==len(race_keys) and result['qualified_team_races']==len(team_races)
    assert result['gate']==(len(race_keys)>=12 and len(team_races)>=60)
    assert not result['response_fit'] and not result['winner_fit']


if __name__=='__main__':
    test_artifacts()
    print('E237 independent all available raw episode/windows/tyres/positions/set-pair support/gate PASS')
