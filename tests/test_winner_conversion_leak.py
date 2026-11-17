"""E186 past residual history is causal; its current pace scores are hindsight."""
import numpy as np
import pandas as pd
from experiments.e186_winner_conversion import conversion_trace,corrected_probability,historical_residual,record_residual


def test_math():
    base=np.array([.6,.3,.1])
    np.testing.assert_allclose(corrected_probability(base,np.zeros(3)),base,atol=1e-15,rtol=0)
    assert historical_residual({},'a',10)==0
    state={}; record_residual(state,'a',0,.4)
    assert np.isclose(historical_residual(state,'a',0),.4/21)
    assert np.isclose(historical_residual(state,'a',10),.2/20.5)
    record_residual(state,'a',10,.2)
    assert np.isclose(historical_residual(state,'a',10),.4/21.5)
    p=corrected_probability(base,np.array([-99.,4.,.1]))
    assert (p>0).all() and np.isclose(p.sum(),1)


def test_source_cutoff():
    df=pd.read_csv('data/results.csv').loc[lambda d:d.season.eq(2019)&d['round'].le(6)].copy()
    pace=pd.read_csv('data/racepace_ff1.csv').loc[lambda d:d.season.eq(2019)&d['round'].le(6)].copy()
    before=conversion_trace(df,pace)
    bad=df.copy(); altered=pace.copy(); rng=np.random.default_rng(18)
    for _,ids in bad[bad['round'].ge(4)].groupby('round').groups.items():
        bad.loc[ids,'order']=rng.permutation(np.arange(1,len(ids)+1))
    altered.loc[altered['round'].ge(4),'pace_gap']=rng.uniform(0,9,altered['round'].ge(4).sum())
    labels_only=conversion_trace(bad,pace)
    diagnostic_columns=['base_probability','corrected_probability','null_probability']
    pd.testing.assert_frame_equal(before.loc[before['round'].eq(4),diagnostic_columns],labels_only.loc[labels_only['round'].eq(4),diagnostic_columns],check_exact=True)
    after=conversion_trace(bad,altered)
    columns=['prior_driver_residual','prior_null_residual']
    target=before['round'].eq(4)
    pd.testing.assert_frame_equal(before.loc[target,columns],after.loc[target,columns],check_exact=True)
    assert not before.loc[before['round'].eq(6),columns].equals(after.loc[after['round'].eq(6),columns])
    assert not before.loc[target,'base_probability'].equals(after.loc[target,'base_probability'])
    np.testing.assert_allclose(before.groupby(['season','round']).base_probability.sum(),1)
    first=before['round'].eq(1)
    np.testing.assert_allclose(before.loc[first,'base_probability'],before.loc[first,'corrected_probability'],atol=1e-15,rtol=0)
    missing=pace.copy(); missing.loc[missing['round'].eq(1),'pace_gap']=np.nan
    missing_trace=conversion_trace(df,missing)
    assert not missing_trace[missing_trace['round'].eq(1)].eligible.any()
    assert missing_trace[missing_trace['round'].eq(2)].prior_driver_residual.eq(0).all()
    shuffled=conversion_trace(df.sample(frac=1,random_state=8),pace.sample(frac=1,random_state=9))
    pd.testing.assert_frame_equal(before,shuffled,check_exact=True)


if __name__=='__main__':
    test_math()
    test_source_cutoff()
    print('conversion shrinkage, normalization, missing opportunities, cutoff and hindsight distinction passed')
