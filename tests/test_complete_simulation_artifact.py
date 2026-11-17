"""E193 same-draw and full-denominator audit, preserving E192 artifacts."""
import numpy as np
import pandas as pd


def test_complete_artifacts():
    p=pd.read_csv('results/e193_inner_predictions.csv',float_precision='round_trip')
    s=pd.read_csv('results/e193_inner_simulation_probabilities.csv',float_precision='round_trip')
    r=pd.read_csv('results/e193_inner_per_race.csv')
    reduced=pd.read_csv('results/e192_inner_predictions.csv',float_precision='round_trip')
    keys=['season','round','driver','model']
    for model in ('standings','pre_logit','win_logit','win_null','ens'):
        pd.testing.assert_frame_equal(p[p.model.eq(model)].sort_values(keys).reset_index(drop=True),reduced[reduced.model.eq(model)].sort_values(keys).reset_index(drop=True),check_exact=True)
    assert r.groupby('model').size().eq(61).all() and r.season.max()==2022
    assert not s.duplicated(['season','round','driver','null']).any()
    for (season,round_,null),g in s.groupby(['season','round','null'],sort=True):
        assert g.effective_draws.eq(4000).all()
        win=g.win_probability.to_numpy(); pod=g.podium_probability.to_numpy()
        assert np.isclose(win.sum(),1) and np.isclose(pod.sum(),3)
        assert (win<=pod).all()
        np.testing.assert_allclose(win*4000,np.round(win*4000),atol=1e-10)
        np.testing.assert_allclose(pod*4000,np.round(pod*4000),atol=1e-10)
        np.testing.assert_allclose(g.smooth_win,(win*4000+.5)/(4000+.5*len(g)),atol=1e-14)
        modes=[('mc_null','win_probability')] if null else [('mc_win','win_probability'),('mc_podium','podium_probability')]
        for model,col in modes:
            score=p[p.season.eq(season)&p['round'].eq(round_)&p.model.eq(model)].set_index('driver').score
            np.testing.assert_array_equal(score.reindex(g.driver),-g[col])
    official=pd.read_csv('data/results.csv').loc[lambda d:d.season.between(2020,2022),['season','round','driver','team','order']]
    actual=official[official.order.eq(1)].set_index(['season','round'])
    chosen=p.sort_values(['season','round','model','score','driver']).drop_duplicates(['season','round','model']).merge(official,on=['season','round','driver'],validate='many_to_one')
    for model,count in (('mc_win',38),('mc_podium',36),('mc_null',8)):
        g=chosen[chosen.model.eq(model)].set_index(['season','round'])
        assert g.team.eq(actual.team).sum()==count
    checked=s.merge(official,on=['season','round','driver'],validate='many_to_one')
    loss=checked[checked.order.eq(1)].assign(loss=lambda d:-np.log(d.smooth_win)).groupby('null').loss.mean()
    assert np.isclose(loss.loc[False],1.6000665994306849)
    assert np.isclose(loss.loc[True],5.5562436274315505)
    wins=r.groupby('model').winner.sum()
    assert wins.mc_win==32 and wins.mc_podium==31
    baseline=pd.read_csv('results/e192_inner_per_race.csv').pivot(index=['season','round'],columns='model',values='winner')
    current=r.pivot(index=['season','round'],columns='model',values='winner')
    print('full versus reduced simulated-winner hit/miss changes:',current.mc_win.ne(baseline.mc_win).sum())


if __name__=='__main__':
    test_complete_artifacts()
    print('E193 complete simulation probabilities, scores, team hits, loss and unchanged references reproduce')
