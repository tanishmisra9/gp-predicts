"""Thermal support fails on a confounded trend and ignores response labels."""
import numpy as np
import pandas as pd

from experiments.e198_temperature_support import design,measure,prepare,support


def test_support():
    n=80; phase=np.linspace(0,1,n)
    g=pd.DataFrame(dict(phase=phase,TyreLife=np.arange(n)+1.,Position=1.,close=False,Stint=1.,Compound='SOFT',TrackTemp=30+10*phase+2*phase**2,weather_time=pd.to_timedelta(np.arange(n)*60,unit='s')))
    stats,residual=support(g)
    assert not stats['support'] and stats['residual_rms']<1e-10
    Z=design(g); noise=np.random.default_rng(198).normal(size=n)
    noise-=Z@np.linalg.lstsq(Z,noise,rcond=1e-10)[0]
    noise/=np.sqrt(np.mean(noise**2))
    g.TrackTemp=30+phase+noise
    stats,residual=support(g)
    np.testing.assert_allclose(residual,noise,atol=1e-10)
    assert stats['support'] and np.isclose(stats['residual_rms'],1.)
    np.testing.assert_allclose(Z.T@residual,0.,atol=1e-8)
    changed=g.assign(seconds=np.random.default_rng(8).normal(size=n),order=99)
    assert support(changed)[0]==stats
    np.testing.assert_array_equal(support(changed)[1],residual)
    g.TrackTemp=35.
    assert not support(g)[0]['support'] and support(g)[0]['variance_fraction']==0.
    assert not support(g.iloc[:0])[0]['support']
    assert not support(g.iloc[:1])[0]['support']
    # Alternating readings are shared blocks, with every pilot event retained.
    frames=[]
    for round_ in (5,12,14):
        part=pd.concat([g.iloc[:20].assign(Driver='a'),g.iloc[:20].assign(Driver='b')],ignore_index=True)
        part=part.assign(season=2019,round=round_,LapNumber=np.tile(np.arange(20)+2,2),LapStartTime=part.weather_time,Rainfall=False)
        frames.append(part)
    source=pd.concat(frames,ignore_index=True)
    ready,excluded=prepare(source)
    assert excluded==0 and ready.groupby(['round','weather_time']).half.nunique().eq(1).all()
    assert ready.groupby(['round','Driver']).half.nunique().eq(2).all()
    empty_result,empty_table,empty_rows=measure(source.assign(Rainfall=True))
    assert not empty_result['gate'] and empty_result['eligible_rows']==0 and empty_result['excluded_rows']==len(source)
    assert len(empty_table)==18 and not empty_table.support.any() and empty_rows.empty
    for bad in (pd.concat([source,source.iloc[:1]]),source.assign(season=2026)):
        try:
            prepare(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('duplicate/non-pilot source accepted')


if __name__=='__main__':
    test_support()
    print('thermal trend/orthogonal/empty/constant support, label invariance, shared blocks and pilot guard passed')
