"""Missing-sample zero filling cannot become measured stationary pit time."""
import numpy as np
import pandas as pd

from experiments.e209_stationary_source import allowed_urls,stationary


def test_source():
    origin=pd.Timestamp('2019-01-01'); seconds=np.arange(0,25,.25)
    car=pd.DataFrame(dict(Date=origin+pd.to_timedelta(seconds,unit='s'),
        Time=pd.to_timedelta(seconds,unit='s'),Speed=np.where((seconds>=10)&(seconds<=12),0,60)))
    result=stationary(car,0,24,origin)
    assert result['usable'] and result['samples']==9
    assert result['lower']==2 and result['upper']==2.5 and result['midpoint']==2.25
    padded=car.copy(); padded.loc[padded.Speed.eq(0),'Time']=pd.NaT
    assert not stationary(padded,0,24,origin)['usable']
    split=car.copy(); split.loc[split.Time.eq(pd.Timedelta(seconds=11)),'Time']=pd.NaT
    assert not stationary(split,0,24,origin)['usable']
    two=car.copy(); two.loc[two.Time.between(pd.Timedelta(seconds=3),pd.Timedelta(seconds=5)),'Speed']=0
    assert stationary(two,0,24,origin)['candidates']==2
    assert not stationary(two,0,24,origin)['usable']
    missing=car.copy(); missing.loc[missing.Time.eq(pd.Timedelta(seconds=9.75)),'Time']=pd.NaT
    assert not stationary(missing,0,24,origin)['usable']
    later=car.copy(); later.loc[later.Time.gt(pd.Timedelta(seconds=20)),'Speed']=0
    assert stationary(later,0,15,origin)==stationary(car,0,15,origin)
    assert not stationary(car.iloc[:0],0,24,origin)['usable']
    urls=allowed_urls('/static/2019/2019-05-12_Spanish_Grand_Prix/2019-05-12_Race/')
    assert len(urls)==2 and all(u.endswith('/CarData.z.jsonStream') for u in urls)
    try:
        allowed_urls('/static/2026/x/')
    except AssertionError:
        pass
    else:
        raise AssertionError('non-pilot-year URL accepted')


if __name__=='__main__':
    test_source()
    print('stationary brackets, padded/split/multiple/missing runs, endpoint invariance and URL guard PASS')
