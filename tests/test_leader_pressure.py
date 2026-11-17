"""Conservative gap age and backward-only leader snapshots."""
import numpy as np
import pandas as pd

from experiments.e199_leader_pressure import contrasting_blocks,exposures,numeric_gap,prepare_stream


def test_pressure():
    for value,expected in [('1.25',1.25),('+2.0',2.),('5',5.)]:
        assert numeric_gap(value)==expected
    for value in ('1L','LAP 10','1:02.5','NaN','inf','-1','1e2','9'*999):
        assert np.isnan(numeric_gap(value))
    identity=pd.DataFrame(dict(DriverNumber=['1','2'],Driver=['AAA','BBB']))
    stream=pd.DataFrame(dict(Driver=['1','2','1','2','2','2','2'],Time=pd.to_timedelta([0,0,40,40,60,61,80],unit='s'),Position=[1,2,1,2,2,2,2],GapToLeader=['LAP 1','1.0','LAP 2','1.0','2.0','5.0','5.0']))
    target=pd.DataFrame(dict(Driver=['AAA']*5,LapNumber=[4,5,6,7,8],LapStartTime=pd.to_timedelta([30,40,60,61,92],unit='s'),Stint=1,Compound='MEDIUM',TyreLife=np.arange(5)+4))
    source=prepare_stream(stream,identity)
    out=exposures(source,target)
    assert list(out.lap)==[4,6,7] # equal-value update at40 cannot refresh a stale gap
    assert list(out.state)==['pressured','pressured','comfortable']
    assert list(out.age)==[30.,0.,0.]
    assert (out.observed<=out.start).all()
    fixed=target[target.LapNumber.eq(6)]
    changed=stream.copy(); changed.loc[changed.Time.gt(pd.Timedelta(seconds=60)),'GapToLeader']='29.5'
    pd.testing.assert_frame_equal(exposures(prepare_stream(changed,identity),fixed),exposures(source,fixed),check_exact=True)
    prior=stream.copy(); prior.loc[prior.Time.eq(pd.Timedelta(seconds=60)),'GapToLeader']='6.0'
    assert exposures(prepare_stream(prior,identity),fixed).state.iloc[0]=='comfortable'
    duplicate=pd.concat([stream,stream.iloc[[4]].assign(GapToLeader='3.0')],ignore_index=True)
    assert exposures(prepare_stream(duplicate,identity),fixed).state.iloc[0]=='intermediate'
    conflict=stream.copy(); conflict.loc[conflict.Driver.eq('1'),'Position']=2
    assert exposures(prepare_stream(conflict,identity),fixed).empty
    try:
        prepare_stream(stream,pd.concat([identity,identity.iloc[:1]]))
    except AssertionError:
        pass
    else:
        raise AssertionError('duplicate identity accepted')
    rows=pd.concat([out.iloc[[0]].assign(state='pressured')]*5+[out.iloc[[0]].assign(state='comfortable')]*5,ignore_index=True)
    rows=rows.assign(season=2019,round=1,team='test')
    assert len(contrasting_blocks(rows))==1
    assert contrasting_blocks(rows.iloc[:9]).empty
    assert contrasting_blocks(rows.iloc[:0]).empty


if __name__=='__main__':
    test_pressure()
    print('gap parsing/change clock, stale/threshold boundaries, duplicates, identities, backward snapshots and blocks passed')
