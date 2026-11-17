"""Fixed first-full-lap contrast rejects pit, neutralization and stint errors."""
import numpy as np
import pandas as pd

from experiments.e204_warmup import blocks,measurements,past_response


def fixture():
    n=8
    x=pd.DataFrame(dict(Driver='a',LapNumber=np.arange(n)+1,LapTime=pd.to_timedelta([90.]*n,unit='s'),
        Time=pd.to_timedelta(np.arange(1,n+1)*90,unit='s'),LapStartTime=pd.to_timedelta(np.arange(n)*90,unit='s'),
        Stint=[1,1,2,2,2,2,2,2],Compound='MEDIUM',TyreLife=[1,2,1,2,3,4,5,6],
        Position=2.,TrackStatus='1',IsAccurate=True,PitInTime=pd.NaT,PitOutTime=pd.NaT))
    x.PitInTime=pd.Series(pd.NaT,index=x.index,dtype='timedelta64[ns]'); x.PitOutTime=x.PitInTime.copy()
    x.loc[1,'PitInTime']=pd.Timedelta(seconds=175)
    x.loc[2,'PitOutTime']=pd.Timedelta(seconds=195)
    return x


def test_blocks():
    x=fixture(); assert len(blocks(x))==1
    assert abs(blocks(x).contrast.iloc[0])<1e-12
    x.loc[3,'LapTime']=pd.Timedelta(seconds=90.6)
    b=blocks(x); assert b.first_full_lap.iloc[0]==4
    np.testing.assert_allclose(b.contrast.iloc[0],.6/90.15*100,atol=1e-12)
    changed=x.assign(order=999,grid=-1,points=0)
    pd.testing.assert_frame_equal(blocks(changed),b)
    linear=fixture(); linear.loc[3:6,'LapTime']=pd.to_timedelta([93,92,91,90],unit='s')
    assert abs(blocks(linear).contrast.iloc[0])<1e-12
    for col,value in [('TrackStatus','4'),('IsAccurate',False),('Stint',3),('Position',3.),('TyreLife',9.)]:
        bad=x.copy(); bad.loc[5,col]=value
        assert blocks(bad).empty
    assert blocks(x.drop(index=5)).empty
    assert blocks(x.assign(PitInTime=pd.NaT)).empty
    assert blocks(x.iloc[:0]).empty
    try:
        blocks(pd.concat([x,x.iloc[:1]]))
    except AssertionError:
        pass
    else:
        raise AssertionError('duplicate lap accepted')
    entries=pd.DataFrame([dict(season=2019,round=r,team=f't{t}',code=d)
        for r in range(1,7) for t in range(6) for d in (f'a{t}',f'b{t}')])
    source=pd.DataFrame([dict(season=2019,round=r,team=f't{t}',driver=d,contrast=float(t))
        for r in range(1,7) for t in range(6) for d in (f'a{t}',f'b{t}')])
    table=measurements(source,entries)
    assert table.centered_response.notna().all()
    forecast=past_response(table)
    changed=table.copy(); changed.loc[changed['round'].ge(4),'centered_response']=999
    again=past_response(changed)
    np.testing.assert_array_equal(forecast.loc[forecast['round'].le(4),'prediction'],again.loc[again['round'].le(4),'prediction'])
    earlier=table.copy(); earlier.loc[earlier['round'].eq(1),'centered_response']=999
    assert not past_response(earlier).prediction.equals(forecast.prediction)


if __name__=='__main__':
    test_blocks()
    print('warm-up contrast/trend, invalid blocks, metadata invariance and strict prior forecasts PASS')
