"""Reconstruct E188 winning-team choices and metrics from saved forecasts."""
import numpy as np
import pandas as pd


def test_saved_fusion():
    r=pd.read_csv('results/e188_inner_per_race.csv')
    f=pd.read_csv('results/e188_inner_team_forecasts.csv',float_precision='round_trip')
    p=pd.read_csv('results/e181_inner_predictions.csv',float_precision='round_trip')
    official=pd.read_csv('data/results.csv').loc[lambda d:d.season.between(2020,2022)]
    assert len(r)==61*6 and r.groupby('model').size().eq(61).all()
    assert r.season.max()==2022 and f.season.min()==2019 and f.season.max()==2022
    assert not f.duplicated(['season','round','model','team']).any()
    for row in r.itertuples():
        race=official[official.season.eq(row.season)&official['round'].eq(row.round)]
        actual=race[race.order.eq(1)].iloc[0]
        selected=[row.pred1,row.pred2,row.pred3]
        assert len(set(selected))==3 and set(selected).issubset(set(race.driver))
        assert row.winner==float(row.pred1==actual.driver)
        assert np.isclose(row.podium,len(set(selected)&set(race[race.order.le(3)].driver))/3)
        assert row.team_hit==float(race.set_index('driver').loc[row.pred1,'team']==actual.team)
        base=p[p.season.eq(row.season)&p['round'].eq(row.round)&p.model.eq('win_logit')].sort_values(['score','driver'])
        if row.model.startswith('fusion_'):
            strength=f[f.season.eq(row.season)&f['round'].eq(row.round)&f.model.eq(row.model)]
            assert set(strength.team)==set(race.team)
            team=strength.sort_values(['strength','team']).team.iloc[0]
            ordered=base.merge(race[['driver','team']],on='driver',validate='one_to_one')
            winner=ordered[ordered.team.eq(team)].driver.iloc[0]
            expected=[winner]+[driver for driver in ordered.driver if driver!=winner]
        else:
            expected=p[p.season.eq(row.season)&p['round'].eq(row.round)&p.model.eq(row.model)].sort_values(['score','driver']).driver.tolist()
        assert selected==expected[:3]
    old=pd.read_csv('results/e181_inner_per_race.csv').set_index(['season','round','model'])
    baseline=r[r.model.isin(['standings','ens','win_logit'])].set_index(['season','round','model'])
    np.testing.assert_allclose(baseline[['winner','podium']],old.loc[baseline.index,['winner','podium']])
    wins=r.groupby('model').winner.sum(); teams=r.groupby('model').team_hit.sum()
    assert wins.fusion_equal==28 and teams.fusion_equal==34
    assert wins.fusion_single==25 and teams.fusion_single==31


if __name__=='__main__':
    test_saved_fusion()
    print('E188 full denominator, saved winning-team/driver choice and all metrics reproduce')
