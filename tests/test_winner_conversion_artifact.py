"""Independent checks of E186 hindsight probabilities and all-race metrics."""
import numpy as np
import pandas as pd


def test_saved_conversion():
    t=pd.read_csv('results/e186_inner_conversion_trace.csv',float_precision='round_trip')
    r=pd.read_csv('results/e186_inner_hindsight_metrics.csv')
    official=pd.read_csv('data/results.csv').loc[lambda d:d.season.between(2020,2022),['season','round','driver','team','order']]
    assert not t.duplicated(['season','round','driver']).any()
    assert len(t)==len(official) and t.season.max()==2022
    merged=t[['season','round','driver','team','order']].merge(official,on=['season','round','driver'],validate='one_to_one',suffixes=('_trace','_official'))
    np.testing.assert_array_equal(merged.order_trace,merged.order_official)
    np.testing.assert_array_equal(merged.team_trace,merged.team_official)
    assert r.groupby('model').size().eq(61).all()
    modes={'HINDSIGHT_pace_base':'base_probability','HINDSIGHT_driver_conversion':'corrected_probability','HINDSIGHT_null_history':'null_probability'}
    for key,g in t.groupby(['season','round'],sort=True):
        actual=g[g.order.eq(1)].iloc[0]
        for model,col in modes.items():
            p=g[col].to_numpy()
            assert np.isfinite(p).all() and (p>0).all() and np.isclose(p.sum(),1.)
            chosen=g.sort_values([col,'driver'],ascending=[False,True]).iloc[:3]
            row=r[r.season.eq(key[0])&r['round'].eq(key[1])&r.model.eq(model)].iloc[0]
            assert [row.pred1,row.pred2,row.pred3]==list(chosen.driver)
            assert row.winner==float(row.pred1==actual.driver)
            assert row.team_hit==float(chosen.iloc[0].team==actual.team)
            assert np.isclose(row.podium,len(set(chosen.driver)&set(g[g.order.le(3)].driver))/3)
            assert np.isclose(row.logloss,-np.log(actual[col]))
        expected=np.maximum(np.minimum(g.base_probability+g.prior_driver_residual,1.),1e-6)
        np.testing.assert_allclose(g.corrected_probability,expected/expected.sum(),atol=1e-14)
    eligible=t[t.eligible]
    a=eligible.prior_driver_residual.to_numpy(); b=eligible.residual.to_numpy()
    ac=a-a.mean(); bc=b-b.mean()
    correlation=np.dot(ac,bc)/np.sqrt(np.dot(ac,ac)*np.dot(bc,bc))
    assert len(eligible)==180 and np.isclose(correlation,.3494254056571645)
    names=r.pivot(index=['season','round'],columns='model',values='pred1')
    print('changed corrected winner names',names.HINDSIGHT_driver_conversion.ne(names.HINDSIGHT_pace_base).sum())
    hits=r.pivot(index=['season','round'],columns='model',values='winner')
    np.testing.assert_array_equal(hits.HINDSIGHT_driver_conversion,hits.HINDSIGHT_pace_base)
    assert hits.HINDSIGHT_driver_conversion.sum()==44


if __name__=='__main__':
    test_saved_conversion()
    print('E186 all-race official labels, probability math, winner/overlap/team/loss and correlation checks passed')
