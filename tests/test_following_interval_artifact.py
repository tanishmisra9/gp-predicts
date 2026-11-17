"""E200 independent cached interval clock/as-of, legacy proxies and source counts."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.e190_traffic_feasibility import cached_session
from experiments.e199_leader_pressure import prepare_stream
from experiments.e200_following_interval import following
from f1.data import load_results


def test_artifacts():
    import fastf1
    import f1.ff1
    from fastf1 import _api
    fastf1.Cache.offline_mode(True)
    result=json.loads(Path('results/e200_following_interval_summary.json').read_text())
    rows=pd.read_csv('results/e200_following_interval_exposures.csv',float_precision='round_trip',dtype={'stream_close':'boolean'})
    pairs=pd.read_csv('results/e200_following_interval_pairs.csv',float_precision='round_trip')
    support=pd.read_csv('results/e200_following_interval_support.csv',float_precision='round_trip')
    for col in ('LapStartTime','stream_time','interval_observed'):
        rows[col]=pd.to_timedelta(rows[col])
    pairs['completion2']=pd.to_timedelta(pairs.completion2)
    entries=load_results().loc[lambda d:d.season.eq(2019),['round','code','team']]
    assert [(r['season'],r['round']) for r in result['sessions']]==[(2019,r) for r in range(1,22)]
    assert rows.season.eq(2019).all() and not rows.duplicated(['round','Driver','LapNumber']).any()
    assert 'seconds' not in rows and 'order' not in rows and len(support)==210
    expected_pairs=[]; expected_confusion={name:{key:0 for key in ('00','01','10','11')} for name in ('proxy_close','end_proxy_close')}
    for report in result['sessions']:
        r=report['round']; assert report['available']
        session=cached_session(fastf1,2019,r); raw=pd.DataFrame(session.laps)
        with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('HTTP forbidden in audit')):
            _,stream,_=_api._extended_timing_data(session.api_path)
        ids=raw[['DriverNumber','Driver']].drop_duplicates(); mapping=dict(zip(ids.DriverNumber.astype(str),ids.Driver))
        histories={}
        for number,g in stream.groupby('Driver',sort=True):
            g=g.drop_duplicates('Time',keep='last').sort_values('Time',kind='stable')
            t=g.Time.dt.total_seconds().to_numpy(); gap=pd.to_numeric(g.IntervalToPositionAhead,errors='coerce').to_numpy(float)
            known=np.isfinite(gap); previous=np.r_[np.nan,gap[:-1]]
            changed=known&(~np.isfinite(previous)|(gap!=previous))
            last=np.maximum.accumulate(np.where(changed,np.arange(len(gap)),-1))
            observed=np.where(known&(last>=0),t[np.maximum(last,0)],np.nan)
            histories[mapping[str(number)]]=(t,g.Position.to_numpy(),gap,observed)
        mask=raw.LapNumber.gt(1)&raw.LapTime.notna()&raw.Time.notna()&raw.LapStartTime.notna()&raw.PitInTime.isna()&raw.PitOutTime.isna()&raw.TrackStatus.astype(str).eq('1')&raw.IsAccurate.fillna(False)
        clean=raw[mask].copy(); seconds=clean.LapTime.dt.total_seconds()
        clean=clean[seconds.lt(1.07*seconds.groupby(clean.Driver).transform('median'))].sort_values(['Time','Driver'])
        times=clean.Time.dt.total_seconds().to_numpy(); drivers=clean.Driver.to_numpy()
        end_flags={}; previous=None
        line=raw[raw.Time.notna()&raw.Position.notna()]
        for item in sorted(line.itertuples(),key=lambda x:(x.LapNumber,x.Position)):
            flag=False
            if previous is not None and previous.LapNumber==item.LapNumber:
                flag=0<=(item.Time-previous.Time).total_seconds()<=1.5
            end_flags[(item.Driver,item.LapNumber)]=flag; previous=item
        current=rows[rows['round'].eq(r)].set_index(['Driver','LapNumber'])
        targets=clean[clean.Position.gt(1)]
        assert len(current)==len(targets)==report['targets']
        validated=[]
        for item in targets.itertuples():
            actual=current.loc[(item.Driver,item.LapNumber)]; start=item.LapStartTime.total_seconds()
            a=np.searchsorted(times,start-1.5,'left'); b=np.searchsorted(times,start,'right')
            old=bool(np.any(drivers[a:b]!=item.Driver))
            assert actual.proxy_close==old and actual.end_proxy_close==end_flags[(item.Driver,item.LapNumber)]
            t,position,gap,observed=histories[item.Driver]; i=np.searchsorted(t,start,'right')-1
            assert i>=0
            age=start-observed[i]
            fresh=bool(np.isfinite(gap[i]) and gap[i]>0 and position[i]>1 and 0<=age<=30)
            assert actual.fresh==fresh
            np.testing.assert_allclose([actual.stream_time.total_seconds(),actual.live_position,actual.interval,actual.interval_observed.total_seconds(),actual.interval_age],[t[i],position[i],gap[i],observed[i],age],atol=1e-9,equal_nan=True)
            assert actual.LapStartTime==item.LapStartTime and actual.Position==item.Position and actual.Stint==item.Stint and actual.Compound==item.Compound and actual.TyreLife==item.TyreLife
            if fresh:
                close=gap[i]<=1.5; assert actual.stream_close==close
                for name,flag in (('proxy_close',old),('end_proxy_close',end_flags[(item.Driver,item.LapNumber)])):
                    expected_confusion[name][f'{int(flag)}{int(close)}']+=1
                validated.append((item.Driver,item.LapNumber,item.Stint,item.Compound,item.TyreLife,item.Position,close,item.Time))
            else:
                assert pd.isna(actual.stream_close)
        assert len(validated)==report['matched'] and np.isclose(len(validated)/len(targets),report['coverage'])
        by_driver={}
        for item in validated:
            by_driver.setdefault(item[0],[]).append(item)
        event_pairs=[]
        for driver,values in sorted(by_driver.items()):
            values=sorted(values,key=lambda x:x[1]); used=set()
            for first,second in zip(values[:-1],values[1:]):
                if first[1] in used or second[1] in used:
                    continue
                if second[1]==first[1]+1 and first[2:4]==second[2:4] and np.isclose(second[4]-first[4],1.) and first[5]==second[5] and first[6]!=second[6]:
                    team=entries[entries['round'].eq(r)&entries.code.eq(driver)].team.iloc[0]
                    event_pairs.append((2019,r,driver,first[1],second[1],bool(second[6]),first[2],second[7],team))
                    used.update((first[1],second[1]))
        assert len(event_pairs)==report['pairs']; expected_pairs.extend(event_pairs)
        if r==12:
            selected=current[current.fresh].iloc[len(current[current.fresh])//2]; driver,lap=selected.name
            fixed=targets[targets.Driver.eq(driver)&targets.LapNumber.eq(lap)]
            source=prepare_stream(stream,ids,field='IntervalToPositionAhead'); before=following(source,fixed)
            altered=stream.copy(); altered.loc[altered.Time.gt(selected.LapStartTime),'IntervalToPositionAhead']='0.25'
            pd.testing.assert_frame_equal(before,following(prepare_stream(altered,ids,field='IntervalToPositionAhead'),fixed),check_exact=True)
            number=ids.set_index('Driver').loc[driver,'DriverNumber']
            possible=altered[altered.Driver.astype(str).eq(str(number))&altered.Time.le(selected.LapStartTime)]
            last_index=possible.sort_values('Time',kind='stable').index[-1]
            altered.loc[last_index,'IntervalToPositionAhead']='2.5' if selected.stream_close else '0.5'
            assert not before.equals(following(prepare_stream(altered,ids,field='IntervalToPositionAhead'),fixed))
    actual_pairs=list(pairs[['season','round','driver','lap1','lap2','entering','stint','completion2','team']].itertuples(index=False,name=None))
    assert actual_pairs==expected_pairs
    assert result['confusion']==expected_confusion
    basic={}
    for (r,team),_ in entries.groupby(['round','team']):
        group=sorted([p for p in expected_pairs if p[1]==r and p[8]==team],key=lambda p:(p[7],p[2],p[3]))
        states=[p[5] for p in group]; enter=sum(states); leave=len(states)-enter
        halves=[states[0::2],states[1::2]]
        half_ok=[len(h)>=4 and any(h) and not all(h) for h in halves]
        basic[(r,team)]=dict(pairs=len(group),entering=enter,leaving=leave,full=len(group)>=8 and enter>=2 and leave>=2,half0=half_ok[0],half1=half_ok[1])
    counts={r:{name:sum(q[name] for (round_,_),q in basic.items() if round_==r) for name in ('full','half0','half1')} for r in range(1,22)}
    past={}; valid_races=set(); half_total=temporal=0
    for actual in support.sort_values(['round','team']).itertuples():
        key=(actual.round,actual.team); expected=basic[key]
        for name,value in expected.items():
            assert getattr(actual,name)==value
        valid=expected['full'] and counts[actual.round]['full']>=6
        halves=all(expected[name] and counts[actual.round][name]>=6 for name in ('half0','half1'))
        earlier=past.get(actual.team,0)
        assert actual.valid_full==valid and actual.valid_halves==halves and actual.past_full==earlier
        if valid:
            valid_races.add(actual.round); temporal+=int(earlier>=3); past[actual.team]=earlier+1
        half_total+=int(halves)
    covered=sum(r['coverage']>=.8 for r in result['sessions']); primary=expected_confusion['proxy_close']
    disagree=(primary['01']+primary['10'])/int(rows.fresh.sum())
    gate=covered>=15 and rows.fresh.sum()>=10000 and disagree>=.05 and primary['01']>=100 and primary['10']>=100 and len(valid_races)>=12 and half_total>=60 and temporal>=60
    assert result['targets']==len(rows) and result['matched']==rows.fresh.sum() and result['covered_races']==covered
    assert np.isclose(result['disagreement'],disagree) and result['pairs']==len(pairs)
    assert result['valid_response_races']==len(valid_races) and result['half_opportunities']==half_total and result['temporal_opportunities']==temporal and result['gate']==gate


if __name__=='__main__':
    test_artifacts()
    print('E200 all21 interval clocks/as-of, both proxies, pair greediness/disjointness, source supports/gate and real cutoff reproduce')
