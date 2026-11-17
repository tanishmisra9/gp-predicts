"""Same-compound ideal sectors and past-only team histories."""
import numpy as np
import pandas as pd

from experiments.e208_ideal_q1 import NEW,add_history,measure


def fixture():
    rows=[]
    for i in range(10):
        sectors=[30+i/10,30.,30.]
        rows.append(dict(Driver=f'd{i}',Compound='SOFT',LapTime=sum(sectors),
            Sector1Time=sectors[0],Sector2Time=sectors[1],Sector3Time=sectors[2],
            PitInTime=pd.NaT,PitOutTime=pd.NaT,TrackStatus='1',IsAccurate=True,Deleted=False))
    extra=rows[0].copy(); extra.update(Sector1Time=29.,Sector2Time=32.,LapTime=91.)
    rows.append(extra)
    extra=rows[0].copy(); extra.update(Compound='MEDIUM',Sector1Time=28.,Sector2Time=28.,Sector3Time=40.,LapTime=96.)
    rows.append(extra)
    frame=pd.DataFrame(rows)
    frame=frame.assign(LapNumber=frame.groupby('Driver').cumcount()+1,PitInTime=pd.NaT,PitOutTime=pd.NaT,TrackStatus='1',IsAccurate=True,Deleted=False)
    for c in ('LapTime','Sector1Time','Sector2Time','Sector3Time'):
        frame[c]=pd.to_timedelta(frame[c],unit='s')
    return frame


def test_measurement():
    x=fixture(); output=measure(x); a=output.set_index('code').loc['d0']
    assert a.ideal_seconds==89 and a.best_seconds==90
    assert output.ideal_relative.notna().all() and abs(output.ideal_relative.median())<1e-12
    pd.testing.assert_frame_equal(measure(x.assign(order=99,grid=-1)),output)
    assert measure(x[x.Driver.ne('d9')]).ideal_relative.isna().all()
    assert measure(x.iloc[:0]).empty
    for name,value in [('TrackStatus','4'),('IsAccurate',False),('Compound','WET'),('Deleted',True)]:
        bad=x.copy(); bad.loc[0,name]=value
        assert measure(bad).set_index('code').loc['d0','ideal_seconds']==91
    bad=x.copy(); bad.loc[0,'Sector2Time']=pd.Timedelta(seconds=35)
    assert measure(bad).set_index('code').loc['d0','ideal_seconds']==91


def test_history():
    X=pd.DataFrame([dict(season=2018,round=k+1,k=k,code=d,team=t)
        for k in range(6) for d,t in [('a','one'),('b','one'),('c','two')]])
    source=X[['season','round','code']].assign(ideal_relative=1.,actual_relative=2.)
    original=add_history(X,source)
    np.testing.assert_allclose(original.loc[X['round'].eq(2),NEW[0]],.5**.1/(5+.5**.1))
    changed=source.copy(); changed.loc[changed['round'].ge(4),['ideal_relative','actual_relative']]=999
    later=add_history(X,changed)
    np.testing.assert_array_equal(original.loc[X['round'].le(4),NEW],later.loc[X['round'].le(4),NEW])
    earlier=source.copy(); earlier.loc[earlier['round'].eq(1),['ideal_relative','actual_relative']]=5
    assert not add_history(X,earlier).loc[X['round'].eq(4),NEW].equals(original.loc[X['round'].eq(4),NEW])
    assert add_history(X,source.iloc[:0])[NEW].eq(0).all().all()
    assert add_history(X,source.assign(ideal_relative=np.nan,actual_relative=np.nan))[NEW].eq(0).all().all()


if __name__=='__main__':
    test_measurement(); test_history()
    print('same-compound ideal sectors, quality/coverage exclusions and strict prior team history PASS')
