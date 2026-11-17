"""E199 independent array-based as-of reconstruction from all cached streams."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.e190_traffic_feasibility import cached_session
from experiments.e199_leader_pressure import exposures,prepare_stream


def test_artifacts():
    import fastf1
    import f1.ff1
    from fastf1 import _api
    fastf1.Cache.offline_mode(True)
    result=json.loads(Path('results/e199_leader_pressure_summary.json').read_text())
    saved=pd.read_csv('results/e199_leader_pressure_exposures.csv',float_precision='round_trip')
    blocks=pd.read_csv('results/e199_leader_pressure_blocks.csv',float_precision='round_trip')
    for col in ('start','observed'):
        saved[col]=pd.to_timedelta(saved[col])
    assert [(r['season'],r['round']) for r in result['sessions']]==[(2019,r) for r in range(1,22)]
    assert not saved.duplicated(['season','round','driver','lap']).any()
    assert 'seconds' not in saved and saved.season.eq(2019).all()
    for report in result['sessions']:
        round_=report['round']; assert report['available']
        session=cached_session(fastf1,2019,round_); raw=pd.DataFrame(session.laps)
        with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('HTTP forbidden during artifact audit')):
            _,stream,_=_api._extended_timing_data(session.api_path)
        assert len(stream)==report['stream_rows']
        assert int(stream.duplicated(['Driver','Time']).sum())==report['duplicate_stream_keys']
        identities=raw[['DriverNumber','Driver']].drop_duplicates()
        names=dict(zip(identities.DriverNumber.astype(str),identities.Driver))
        histories={}; unknown=0
        for number,g in stream.groupby('Driver',sort=True):
            g=g.drop_duplicates('Time',keep='last').sort_values('Time',kind='stable')
            t=g.Time.dt.total_seconds().to_numpy()
            x=pd.to_numeric(g.GapToLeader,errors='coerce').to_numpy(float)
            assert np.all(np.isfinite(x)|np.isnan(x))
            known=np.isfinite(x); unknown+=int((~known).sum())
            previous=np.r_[np.nan,x[:-1]]
            changed=known&(~np.isfinite(previous)|(x!=previous))
            last=np.maximum.accumulate(np.where(changed,np.arange(len(x)),-1))
            observed=np.where((last>=0)&known,t[np.maximum(last,0)],np.nan)
            histories[names[str(number)]]=(t,g.Position.to_numpy(),x,observed)
        assert unknown==report['unknown_gap_rows']
        mask=raw.LapNumber.gt(1)&raw.LapTime.notna()&raw.Time.notna()&raw.LapStartTime.notna()&raw.PitInTime.isna()&raw.PitOutTime.isna()&raw.TrackStatus.astype(str).eq('1')&raw.IsAccurate.fillna(False)
        clean=raw[mask].copy(); seconds=clean.LapTime.dt.total_seconds()
        med=seconds.groupby(clean.Driver).transform('median')
        clean=clean[seconds.lt(1.07*med)]
        targets=clean[clean.Position.eq(1)&clean.LapNumber.ge(4)]
        assert len(targets)==report['leader_targets']
        expected=[]
        for row in targets.itertuples():
            start=row.LapStartTime.total_seconds(); snapshot=[]
            for driver,(t,position,gap,observed) in histories.items():
                i=np.searchsorted(t,start,side='right')-1
                if i>=0:
                    snapshot.append((driver,position[i],gap[i],observed[i]))
            leaders=[v for v in snapshot if v[1]==1]; pursuers=[v for v in snapshot if v[1]==2]
            if len(leaders)!=1 or len(pursuers)!=1 or leaders[0][0]!=row.Driver:
                continue
            chaser,_,gap,observed=pursuers[0]; age=start-observed
            if not np.isfinite(gap) or not 0<gap<=30 or not 0<=age<=30:
                continue
            expected.append((row.Driver,row.LapNumber,chaser,gap,observed,age,'pressured' if gap<=2 else ('comfortable' if gap>=5 else 'intermediate'),row.Stint,row.Compound,start))
        current=saved[saved['round'].eq(round_)].set_index(['driver','lap'])
        assert len(current)==len(expected)==report['exposures']
        for driver,lap,chaser,gap,observed,age,state,stint,compound,start in expected:
            actual=current.loc[(driver,lap)]
            assert actual.chaser==chaser and actual.state==state and actual.stint==stint and actual.compound==compound
            np.testing.assert_allclose([actual.gap,actual.observed.total_seconds(),actual.age,actual.start.total_seconds()],[gap,observed,age,start],atol=1e-9)
        assert np.isclose(len(current)/len(targets),report['coverage'])
        assert current.state.value_counts().to_dict()==report['states']
        if round_==12:
            sample=current[current.state.eq('pressured')].iloc[len(current[current.state.eq('pressured')])//2]
            key=sample.name
            fixed=targets[targets.Driver.eq(key[0])&targets.LapNumber.eq(key[1])]
            original=exposures(prepare_stream(stream,identities),fixed)
            altered=stream.copy(); altered.loc[altered.Time.gt(sample.start),'GapToLeader']='29.5'
            altered.loc[altered.Time.gt(sample.start),'Position']=99
            pd.testing.assert_frame_equal(original,exposures(prepare_stream(altered,identities),fixed),check_exact=True)
            chaser_number=identities.set_index('Driver').loc[sample.chaser,'DriverNumber']
            indexes=altered.index[altered.Driver.astype(str).eq(str(chaser_number))&altered.Time.le(sample.start)]
            last_index=altered.loc[indexes].sort_values('Time',kind='stable').index[-1]
            altered.loc[last_index,'GapToLeader']='6.25'
            assert not original.equals(exposures(prepare_stream(altered,identities),fixed))
    expected_blocks=[]
    for (season,round_,driver,stint),g in saved.groupby(['season','round','driver','stint'],sort=True):
        qualifies=False
        for _,part in g.groupby(['chaser','compound'],sort=True):
            states=list(part.state)
            qualifies|=states.count('pressured')>=5 and states.count('comfortable')>=5
        if qualifies:
            expected_blocks.append((season,round_,driver,stint,g.team.iloc[0]))
    assert list(blocks.itertuples(index=False,name=None))==expected_blocks
    by_team=blocks.groupby('team').size().to_dict()
    covered=sum(r['coverage']>=.8 for r in result['sessions'])
    gate=covered>=15 and len(blocks)>=20 and blocks['round'].nunique()>=8 and sum(n>=3 for n in by_team.values())>=3
    assert result['covered_races']==covered and result['exposures']==len(saved)
    assert result['contrasting_blocks']==len(blocks) and result['block_races']==blocks['round'].nunique()
    assert result['blocks_by_team']==by_team and result['gate']==gate


if __name__=='__main__':
    test_artifacts()
    print('E199 all21 raw-cache array snapshots, change clocks, exposure metadata/coverage, blocks/gate and real cutoff reproduce')
