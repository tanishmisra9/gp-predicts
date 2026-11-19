"""Wind support must distinguish nuisance-confounded from orthogonal variation."""
import numpy as np
import pandas as pd

from experiments.e203_wind_support import attach,measure,support
from experiments.e198_temperature_support import design


def test_support():
    n=80; t=np.linspace(0,1,n)
    g=pd.DataFrame(dict(phase=t,TyreLife=np.arange(n)+1.,Position=1.,close=False,
        Stint=1.,Compound='SOFT',TrackTemp=30+np.sin(5*t),WindSpeed=2+t,
        weather_time=pd.to_timedelta(np.arange(n)*60,unit='s')))
    assert not support(g)[0]['support']
    Z=np.column_stack([design(g),g.TrackTemp/10])
    noise=np.random.default_rng(203).normal(size=n)
    noise-=Z@np.linalg.lstsq(Z,noise,rcond=1e-10)[0]
    noise/=np.sqrt(np.mean(noise**2))
    g.WindSpeed=10+t+noise
    stats,residual=support(g)
    assert stats['support']
    np.testing.assert_allclose(residual,noise,atol=1e-9)
    assert support(g.assign(WindSpeed=g.TrackTemp))[0]['residual_rms']<1e-9
    assert not support(g.assign(WindSpeed=2.))[0]['support']
    assert not support(g.iloc[:0])[0]['support']
    assert not support(g.iloc[:1])[0]['support']
    frames=[]; feeds=[]
    for round_ in (5,12,14):
        frames.append(g.assign(season=2019,round=round_,Driver='a',LapNumber=np.arange(n)+2,
            LapStartTime=g.weather_time,Rainfall=False))
        feeds.append(g[['weather_time','WindSpeed']].rename(columns={'weather_time':'Time'}).assign(season=2019,round=round_))
    source=pd.concat(frames,ignore_index=True); weather=pd.concat(feeds,ignore_index=True)
    joined=attach(source.drop(columns='WindSpeed'),weather)
    np.testing.assert_array_equal(joined.WindSpeed,source.WindSpeed)
    result,table,rows=measure(joined)
    other,other_table,other_rows=measure(joined.assign(seconds=-999,order=99,LapTime='corrupted'))
    assert result==other
    pd.testing.assert_frame_equal(table,other_table); pd.testing.assert_frame_equal(rows,other_rows)
    missing,missing_table,_=measure(joined.assign(WindSpeed=np.nan))
    assert missing['eligible_rows']==0 and missing['excluded_rows']==len(joined)
    assert len(missing_table)==9 and not missing_table.support.any() and not missing['gate']
    assert measure(joined.assign(WindSpeed=-1))[0]['eligible_rows']==0
    for bad in (pd.concat([source,source.iloc[:1]]),source.assign(season=2026)):
        try:
            measure(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('duplicate/non-pilot accepted')
    try:
        attach(source.drop(columns='WindSpeed'),pd.concat([weather,weather.iloc[:1]]))
    except pd.errors.MergeError:
        pass
    else:
        raise AssertionError('duplicate weather join accepted')


if __name__=='__main__':
    test_support()
    print('wind confounding/orthogonal/constant/empty, join, fixed pilot and label invariance PASS')
