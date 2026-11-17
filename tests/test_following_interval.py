"""E200 field isolation, causal as-of join and source support counts."""
import numpy as np
import pandas as pd

from experiments.e199_leader_pressure import prepare_stream
from experiments.e200_following_interval import count_support,following


def test_interval():
    identities=pd.DataFrame(dict(DriverNumber=['2'],Driver=['BBB']))
    stream=pd.DataFrame(dict(Driver=['2']*4,Time=pd.to_timedelta([0,20,40,60],unit='s'),Position=[2]*4,GapToLeader=['10','9','8','7'],IntervalToPositionAhead=['1.5','1.5','2.0','1L']))
    targets=pd.DataFrame(dict(Driver=['BBB']*5,LapStartTime=pd.to_timedelta([0,30,31,40,60],unit='s')))
    source=prepare_stream(stream,identities,field='IntervalToPositionAhead')
    result=following(source,targets)
    assert list(result.fresh)==[True,True,False,True,False]
    assert result.stream_close.iloc[0] and result.stream_close.iloc[1] and not result.stream_close.iloc[3]
    assert pd.isna(result.stream_close.iloc[2]) and pd.isna(result.stream_close.iloc[4])
    assert result.interval_age.iloc[1]==30 # GapToLeader changes cannot refresh interval age
    other=stream.copy(); other.GapToLeader='999.0'
    pd.testing.assert_frame_equal(result,following(prepare_stream(other,identities,field='IntervalToPositionAhead'),targets),check_exact=True)
    fixed=targets[targets.LapStartTime.le(pd.Timedelta(seconds=40))]
    altered=stream.copy(); altered.loc[altered.Time.gt(pd.Timedelta(seconds=40)),'IntervalToPositionAhead']='0.5'
    pd.testing.assert_frame_equal(following(source,fixed),following(prepare_stream(altered,identities,field='IntervalToPositionAhead'),fixed),check_exact=True)
    altered.loc[altered.Time.eq(pd.Timedelta(seconds=40)),'IntervalToPositionAhead']='0.5'
    assert following(prepare_stream(altered,identities,field='IntervalToPositionAhead'),targets).stream_close.iloc[3]
    pd.testing.assert_frame_equal(prepare_stream(stream,identities),prepare_stream(stream,identities,field='GapToLeader'),check_exact=True)
    assert following(source,targets.iloc[:0]).empty
    entries=pd.DataFrame([dict(round=r,team=str(t)) for r in range(1,6) for t in range(7)])
    rows=[]
    for r in range(1,6):
        for t in range(7):
            for i in range(8):
                # Different teams fail each half: both marginals have6, intersection5.
                entering=(i%4)<2
                if t==0 and i%2==0:
                    entering=True
                if t==1 and i%2==1:
                    entering=True
                rows.append(dict(round=r,team=str(t),driver=str(t),lap1=i*2+1,completion2=i,entering=entering))
    pairs=pd.DataFrame(rows)
    table,races,halves,temporal=count_support(pairs,entries)
    assert races==5 and halves==25 and temporal==14
    assert table.groupby('round').valid_halves.sum().eq(5).all()
    empty,zero_races,zero_halves,zero_temporal=count_support(pairs.iloc[:0],entries)
    assert zero_races==zero_halves==zero_temporal==0 and len(empty)==len(entries)


if __name__=='__main__':
    test_interval()
    print('interval-specific clocks, exact boundaries/missing, field/future invariance, E199 default and support gates passed')
