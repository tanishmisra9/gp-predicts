"""E215 support and features obey strict race-prefix and identity constraints."""
import numpy as np
import pandas as pd

from experiments.e215_teammate_support import BASE,EXTRA,call_labels,forecasts,probability


def test_support():
    targets=pd.DataFrame([dict(season=2019,round=r,k=r-1,team='team',code='AAA',mate='BBB') for r in range(1,5)])
    source=targets.drop(columns='k').assign(gap=[-.2,.4,-.1,.3],n=[10,20,10,30])
    X=forecasts(targets,source)
    assert X.loc[0,'support']==0 and X.loc[0,'prior_gap']==0
    np.testing.assert_allclose(X.loc[1,'support'],10*2**(-1/10))
    np.testing.assert_allclose(X.loc[1,'prior_gap'],-.2)
    altered=source.copy(); altered.loc[altered['round'].ge(3),['gap','n']]=[99.,1000]
    changed=forecasts(targets,altered)
    pd.testing.assert_frame_equal(X[X['round'].le(3)],changed[changed['round'].le(3)])
    prefix=forecasts(targets,source[source['round'].lt(3)])
    pd.testing.assert_frame_equal(X[X['round'].eq(3)],prefix[prefix['round'].eq(3)])
    old=source.copy(); old.loc[0,'n']*=2
    assert forecasts(targets,old).loc[2,'prior_gap']!=X.loc[2,'prior_gap']
    assert forecasts(targets,source.iloc[:0]).support.eq(0).all()
    for bad in [pd.concat([source,source.iloc[:1]]),source.assign(season=2026),source.assign(n=4),source.assign(n=5.5),source.assign(gap=np.inf)]:
        try:
            forecasts(targets,bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('invalid source accepted')
    tr=X.iloc[:2].assign(correct=[1.,0.]); te=X.iloc[2:3]
    np.testing.assert_array_equal(probability(tr,te,BASE+EXTRA),[.5])
    np.testing.assert_array_equal(probability(tr.iloc[:0],te,BASE),[.5])
    labels=call_labels(pd.DataFrame(dict(prior_gap=[0.,0.,-.2,.2,0.,0.],
        observed_gap=[-.1,.1,-.1,.1,0.,-.1],history_events=[1,1,1,1,1,0])))
    np.testing.assert_allclose(labels,[1,0,1,1,np.nan,np.nan])


if __name__=='__main__':
    test_support()
    print('teammate support weighted counts, fixed mean, strict history, source identity and cold-start PASS')
