"""E214 trend removal, ordinary spread, timing, missingness and key rejection."""
import numpy as np
import pandas as pd

from experiments.e214_consistency_winner import NEW,add_history,event_measurements,stint_spreads


def test_measurement():
    n=np.arange(2,18); seconds=90+.2*n
    def laps(y):
        return pd.DataFrame(dict(Driver='AAA',LapNumber=n,LapTime=pd.to_timedelta(y,unit='s'),
            Time=pd.to_timedelta(n*100,unit='s'),LapStartTime=pd.to_timedelta(n*100-y,unit='s'),
            PitInTime=pd.NaT,PitOutTime=pd.NaT,TrackStatus='1',IsAccurate=True,Stint=1,Compound='MEDIUM'))
    pure=stint_spreads(laps(seconds)); assert len(pure)==1 and pure.spread.iloc[0]<1e-12
    varied=stint_spreads(laps(seconds+np.tile([-.3,.3],8)))
    assert varied.spread.iloc[0]>.3
    scaled=stint_spreads(laps((seconds+np.tile([-.3,.3],8))*2))
    np.testing.assert_allclose(varied.spread,scaled.spread,atol=1e-12)
    assert stint_spreads(laps(seconds).iloc[:7]).empty
    assert stint_spreads(laps(seconds).assign(TrackStatus='4')).empty
    assert stint_spreads(laps(seconds).iloc[:0]).empty
    try:
        stint_spreads(pd.concat([laps(seconds),laps(seconds).iloc[:1]]))
    except AssertionError:
        pass
    else:
        raise AssertionError('duplicate laps accepted')


def test_history():
    X=pd.DataFrame([dict(season=2019,round=r,code=f'd{i}',k=r-1) for r in range(1,5) for i in range(10)])
    stints=X[['season','round','code']].assign(stint=1,compound='MEDIUM',laps=12,spread=np.tile(np.arange(10),4))
    source=event_measurements(stints,X)
    out=add_history(X,source)
    assert out.loc[X['round'].eq(1),NEW].eq(0).all().all()
    w=2**(-1/10)
    np.testing.assert_allclose(out.loc[X['round'].eq(2)&X.code.eq('d0'),NEW[0]],-4.5*w/(5+w))
    changed=stints.copy(); changed.loc[changed['round'].ge(3),'spread']=100
    again=add_history(X,event_measurements(changed,X))
    np.testing.assert_array_equal(out.loc[X['round'].le(3),NEW],again.loc[X['round'].le(3),NEW])
    prefix=add_history(X,event_measurements(stints[stints['round'].lt(3)],X))
    np.testing.assert_array_equal(out.loc[X['round'].eq(3),NEW],prefix.loc[X['round'].eq(3),NEW])
    prior=stints.copy(); prior.loc[prior['round'].eq(1)&prior.code.eq('d0'),'spread']+=1
    assert not add_history(X,event_measurements(prior,X))[NEW].equals(out[NEW])
    assert event_measurements(stints[stints.code.ne('d0')],X).centered_spread.isna().all()
    assert add_history(X,source.iloc[:0])[NEW].eq(0).all().all()
    for bad in [pd.concat([source,source.iloc[:1]]),source.assign(season=2026)]:
        try:
            add_history(X,bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('duplicate/unknown source accepted')


if __name__=='__main__':
    test_measurement(); test_history()
    print('consistency trend removal, scale, minimum laps, centering, past-only history and identity guards PASS')
