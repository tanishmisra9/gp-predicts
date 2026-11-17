"""Independent E192 probability/count, score and baseline verification."""
import numpy as np
import pandas as pd


def test_saved_simulations():
    s=pd.read_csv('results/e192_inner_simulation_probabilities.csv',float_precision='round_trip')
    p=pd.read_csv('results/e192_inner_predictions.csv',float_precision='round_trip')
    r=pd.read_csv('results/e192_inner_per_race.csv')
    official=pd.read_csv('data/results.csv').loc[lambda d:d.season.between(2020,2022),['season','round','driver','team','order']]
    assert r.groupby('model').size().eq(61).all() and s.season.max()==2022
    assert not s.duplicated(['season','round','driver','null']).any()
    for (season,round_,null),g in s.groupby(['season','round','null'],sort=True):
        count=g.effective_draws.iloc[0]
        assert g.effective_draws.nunique()==1 and 0<count<=4000
        win=g.win_probability.to_numpy(); podium=g.podium_probability.to_numpy()
        assert np.isclose(win.sum(),1) and np.isclose(podium.sum(),3)
        np.testing.assert_allclose(win*count,np.round(win*count),atol=1e-10)
        np.testing.assert_allclose(podium*count,np.round(podium*count),atol=1e-10)
        assert (win<=podium).all()
        np.testing.assert_allclose(g.smooth_win,(win*count+.5)/(count+.5*len(g)),atol=1e-14)
        mapping=[('mc_null','win_probability')] if null else [('mc_win','win_probability'),('mc_podium','podium_probability')]
        for model,col in mapping:
            saved=p[p.season.eq(season)&p['round'].eq(round_)&p.model.eq(model)].set_index('driver')
            np.testing.assert_array_equal(saved.score.reindex(g.driver).to_numpy(),-g[col].to_numpy())
    keys=['season','round','driver','model']
    old=pd.read_csv('results/e181_inner_predictions.csv',float_precision='round_trip')
    for model in ('standings','ens','pre_logit','win_logit'):
        pd.testing.assert_frame_equal(p[p.model.eq(model)].sort_values(keys).reset_index(drop=True),old[old.model.eq(model)].sort_values(keys).reset_index(drop=True),check_exact=True)
    truth=official[official.order.eq(1)].set_index(['season','round'])
    chosen=p.sort_values(['season','round','model','score','driver']).drop_duplicates(['season','round','model'])
    chosen=chosen.merge(official,on=['season','round','driver'],validate='many_to_one')
    for model,expected_team in (('mc_win',38),('mc_podium',37),('mc_null',8)):
        a=chosen[chosen.model.eq(model)].set_index(['season','round'])
        assert a.team.eq(truth.team).sum()==expected_team
    merged=s.merge(official,on=['season','round','driver'],validate='many_to_one')
    loss=merged[merged.order.eq(1)].assign(loss=lambda d:-np.log(d.smooth_win)).groupby('null').loss.mean()
    assert np.isclose(loss.loc[False],1.597171456302146)
    assert np.isclose(loss.loc[True],5.547657678368226)
    hits=r.pivot(index=['season','round'],columns='model',values='winner')
    assert hits.mc_win.sum()==32 and hits.mc_podium.sum()==32
    print('same-draw changed hit/miss races',hits.mc_win.ne(hits.mc_podium).sum())


if __name__=='__main__':
    test_saved_simulations()
    print('E192 same-draw probability counts, raw scores, baseline equivalence, team hits and logloss reproduce')
