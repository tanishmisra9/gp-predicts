"""E195 signed error history is causal and centering is entry-list based."""
import numpy as np
import pandas as pd

from experiments.e195_pace_bias import bias_predictions


def test_bias():
    source=pd.DataFrame(dict(k=[0,0,1,1,2,2],team=['a','b']*3,bias=[1.,-1.,2.,-2.,3.,-3.]))
    entries=pd.DataFrame(dict(k=[0,0,1,1,1,2,2,3,3],team=['a','b','a','b','new','a','b','a','b']))
    out=bias_predictions(entries,source)
    assert out.loc[out.k.eq(0),'correction'].eq(0).all()
    w=.5**(1/20); expected=w/(8+w)
    assert np.isclose(out.loc[out.k.eq(1)&out.team.eq('a'),'correction'].iloc[0],expected)
    assert out.loc[out.k.eq(1)&out.team.eq('new'),'correction'].iloc[0]==0
    np.testing.assert_allclose(out.groupby('k').correction.mean(),0.,atol=1e-15)
    changed=source.copy(); changed.loc[changed.k.ge(2),'bias']=999.
    after=bias_predictions(entries,changed)
    pd.testing.assert_frame_equal(out.loc[out.k.le(2)],after.loc[after.k.le(2)],check_exact=True)
    assert not out.loc[out.k.eq(3),'correction'].equals(after.loc[after.k.eq(3),'correction'])
    # A current entrant with no measured future pace must still affect centering.
    asymmetric=source.copy(); asymmetric.bias=[2.,0.,2.,0.,2.,0.]
    complete=bias_predictions(entries,asymmetric)
    only_observed=bias_predictions(entries[~(entries.k.eq(1)&entries.team.eq('new'))],asymmetric)
    assert not np.isclose(complete.loc[complete.k.eq(1)&complete.team.eq('a'),'correction'].iloc[0],only_observed.loc[only_observed.k.eq(1)&only_observed.team.eq('a'),'correction'].iloc[0])
    x=bias_predictions(entries,source,shuffled=True)
    pd.testing.assert_frame_equal(x,bias_predictions(entries,source,shuffled=True),check_exact=True)


if __name__=='__main__':
    test_bias()
    print('signed bias shrinkage, no-history/sign, all-entry centering, null and target/future cutoff passed')
