"""E196 backward matching and restricted-network boundary checks."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiments.e196_weather_source import align_weather,allowed_urls,restricted_send


def test_source():
    laps=pd.DataFrame({'LapStartTime':pd.to_timedelta([0,1,120,121,181],unit='s')})
    source=pd.DataFrame({'Time':pd.to_timedelta([0,180],unit='s'),'TrackTemp':[20.,30.],'Rainfall':[False,False]})
    out=align_weather(laps,source)
    np.testing.assert_allclose(out.TrackTemp,[20,20,20,np.nan,30],equal_nan=True)
    changed=pd.concat([source,pd.DataFrame({'Time':pd.to_timedelta([200],unit='s'),'TrackTemp':[99.],'Rainfall':[False]})])
    pd.testing.assert_frame_equal(out,align_weather(laps,changed),check_exact=True)
    missing=align_weather(laps,source.iloc[:0])
    assert missing.TrackTemp.isna().all()
    earlier=source.copy(); earlier.loc[0,'TrackTemp']=40.
    assert align_weather(laps,earlier).TrackTemp.iloc[0]==40.
    try:
        align_weather(laps,pd.concat([source,source.iloc[:1]]))
    except AssertionError:
        pass
    else:
        raise AssertionError('duplicate weather timestamps accepted')
    urls=allowed_urls('/static/2019/test/Race/')
    called=[]
    def send(adapter,request,**kwargs):
        called.append(request.url)
        return 200
    wrapped=restricted_send(send,urls)
    for url in urls:
        assert wrapped(None,SimpleNamespace(url=url))==200
    for url in ('https://livetiming.formula1.com/static/2026/test/WeatherData.jsonStream','https://account.formula1.com/login',next(iter(urls))+'?extra=1'):
        try:
            wrapped(None,SimpleNamespace(url=url))
        except RuntimeError:
            pass
        else:
            raise AssertionError('non-pilot network access accepted')
    assert set(called)==urls


if __name__=='__main__':
    test_source()
    print('weather backward matching/tolerance/empty/future invariance and network allowlist passed')
