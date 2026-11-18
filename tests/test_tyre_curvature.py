"""E216 separates quadratic age response from intercept/linear pace trends."""
import numpy as np
import pandas as pd

from experiments.e216_tyre_curvature import curvature,measurements
from experiments.e191_traffic_repeatability import past_response


def test_curve():
    age=np.arange(4,16,dtype=float); x=(age-age.mean())/10
    g=pd.DataFrame(dict(TyreLife=age,residual=3+2*x+.8*x*x,seconds=100.))
    np.testing.assert_allclose(curvature(g,12),.8,atol=1e-12)
    np.testing.assert_allclose(curvature(g.iloc[::2],6),.8,atol=1e-12)
    np.testing.assert_allclose(curvature(g.iloc[1::2],6),.8,atol=1e-12)
    np.testing.assert_allclose(curvature(g.assign(residual=3+2*x),12),0,atol=1e-12)
    assert np.isnan(curvature(g.iloc[:11],12)) and np.isnan(curvature(g.iloc[:5],5))
    assert np.isnan(curvature(g.assign(TyreLife=4),12))
    assert np.isnan(curvature(g.iloc[:0],12))


def test_measurements():
    entries=pd.DataFrame([dict(season=2019,round=r,team=f't{i}',code=f'd{i}{j}') for r in range(1,6) for i in range(6) for j in range(2)])
    source=entries.rename(columns={'code':'driver'}).drop(columns='team').assign(stint=1,compound='MEDIUM')
    source['response']=np.tile(np.repeat(np.arange(6),2),5)
    source['half0']=source.response; source['half1']=source.response
    table=measurements(source,entries)
    assert table.centered_response.notna().all()
    np.testing.assert_allclose(table[table.team.eq('t0')].centered_response,-2.5)
    original=past_response(table)
    changed=source.copy(); changed.loc[changed['round'].ge(3),['response','half0','half1']]=99
    again=past_response(measurements(changed,entries))
    np.testing.assert_array_equal(original.loc[original['round'].le(3),'prediction'],again.loc[again['round'].le(3),'prediction'])
    bad=source.copy(); bad.loc[bad.driver.eq('d00'),'half0']=np.nan
    assert measurements(bad,entries).loc[lambda d:d.team.eq('t0'),'half0'].isna().all()
    for bad in [pd.concat([source,source.iloc[:1]]),source.assign(season=2026),source.assign(driver='UNKNOWN')]:
        try:
            measurements(bad,entries)
        except AssertionError:
            pass
        else:
            raise AssertionError('invalid measurement identity accepted')


if __name__=='__main__':
    test_curve(); test_measurements()
    print('tyre curvature known quadratic/linear, age/count support, team centering and strict forecast timing PASS')
