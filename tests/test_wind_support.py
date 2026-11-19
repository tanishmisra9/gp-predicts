"""Confounded/independent/constant wind and response-label invariance; synthetic only."""
import numpy as np
import pandas as pd
from experiments.e227_wind_support import nuisance,support,eligible,attach


def test_support():
    n=100;phase=np.linspace(0,1,n)
    g=pd.DataFrame(dict(phase=phase,TyreLife=np.arange(n)+1.,Position=1.,close=False,Stint=1.,Compound='SOFT',
        TrackTemp=30+phase,AirTemp=20+phase,WindDirection=180.,weather_time=pd.to_timedelta(np.arange(n)*60,unit='s'),
        wind_load=20+3*phase+phase**2))
    stats,residual=support(g);assert not stats['support'] and stats['residual_rms']<1e-10
    Z=nuisance(g);noise=np.random.default_rng(227).normal(size=n)
    noise-=Z@np.linalg.lstsq(Z,noise,rcond=1e-10)[0];noise/=np.sqrt(np.mean(noise**2))
    g.wind_load=20+phase+2*noise
    stats,residual=support(g);assert stats['support']
    np.testing.assert_allclose(residual,2*noise,rtol=0,atol=1e-10)
    np.testing.assert_allclose(Z.T@residual,0,rtol=0,atol=1e-8)
    assert support(g.assign(seconds=999,order=999))[0]==stats
    assert not support(g.assign(wind_load=20.))[0]['support']
    assert not support(g.iloc[:0])[0]['support'] and not support(g.iloc[:1])[0]['support']
    pieces=[]
    for round_ in (5,12,14):
        q=g.iloc[:20].copy().assign(season=2019,round=round_,Driver='A',LapNumber=np.arange(20)+1,
            LapStartTime=g.weather_time.iloc[:20].values,Rainfall=False,WindSpeed=2.)
        pieces.append(q)
    frame=pd.concat(pieces,ignore_index=True)
    ready,_,excluded=eligible(frame);assert len(ready)==60 and excluded==0
    assert ready.groupby(['round','weather_time']).half.nunique().eq(1).all()
    assert eligible(frame.assign(WindSpeed=-1))[0].empty
    assert eligible(frame.assign(WindDirection=360))[0].empty
    assert eligible(frame.assign(weather_time=frame.LapStartTime+pd.Timedelta(seconds=1)))[0].empty
    weather=frame[['season','round','weather_time','AirTemp','WindDirection','WindSpeed']].rename(columns={'weather_time':'Time'})
    merged=attach(frame.drop(columns=['AirTemp','WindDirection','WindSpeed']),weather)
    assert len(merged)==60 and merged.WindSpeed.eq(2).all()
    try:attach(frame.drop(columns=['AirTemp','WindDirection','WindSpeed']),pd.concat([weather,weather.iloc[:1]]))
    except AssertionError:pass
    else:raise AssertionError('duplicate weather timestamp accepted')
    for bad in (pd.concat([frame,frame.iloc[:1]],ignore_index=True),frame.assign(season=2026)):
        try:eligible(bad)
        except AssertionError:continue
        raise AssertionError('duplicate/non-pilot source admitted')


if __name__=='__main__':test_support();print('Wind nuisance/support/boundary/label-invariance tests ok')
