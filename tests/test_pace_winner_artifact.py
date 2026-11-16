"""Independently reconstruct E185 metrics from saved selections and source."""
import numpy as np
import pandas as pd

from experiments.e185_pace_winner_audit import source_table


def test_saved_audit():
    r=pd.read_csv('results/e185_inner_measurement_audit.csv')
    truth=pd.read_csv('data/results.csv').loc[lambda d:d.season.between(2020,2022)]
    assert len(r)==61*30 and not r.duplicated(['season','round','model']).any()
    assert r.groupby('model').size().eq(61).all() and r.season.max()==2022
    for row in r.itertuples():
        race=truth[truth.season.eq(row.season)&truth['round'].eq(row.round)]
        podium=set(race[race.order.le(3)].driver)
        actual=race[race.order.eq(1)].iloc[0]
        selected=[row.pred1,row.pred2,row.pred3]
        assert len(set(selected))==3 and set(selected).issubset(set(race.driver))
        assert row.winner==float(row.pred1==actual.driver)
        assert np.isclose(row.podium,len(set(selected)&podium)/3)
        assert row.team_hit==float(race.set_index('driver').loc[row.pred1,'team']==actual.team)
    old=pd.read_csv('results/e181_inner_per_race.csv').set_index(['season','round','model'])
    refs=r[r.model.isin(['standings','ens','win_logit'])].set_index(['season','round','model'])
    np.testing.assert_allclose(refs[['winner','podium']],old.loc[refs.index,['winner','podium']])
    source=source_table()
    for row in r[r.model.str.startswith('HINDSIGHT_')].itertuples():
        col=row.model.removeprefix('HINDSIGHT_')
        race=truth[truth.season.eq(row.season)&truth['round'].eq(row.round)]
        value=race.merge(source[['season','round','code',col]],on=['season','round','code'],how='left',validate='one_to_one')
        actual=value[value.order.eq(1)].iloc[0]
        assert bool(row.winner_source_available)==pd.notna(actual[col])
        pick=value.sort_values([col,'driver'],na_position='last').driver.iloc[0]
        assert pick==row.pred1
    # Separate scalar summation checks actual prior selections without helper group means.
    all_results=pd.read_csv('data/results.csv').loc[lambda d:d.season.le(2022)]
    calendar=all_results[['season','round']].drop_duplicates().sort_values(['season','round'])
    event_to_k={tuple(event):i for i,event in enumerate(calendar.to_numpy())}
    for season,round_ in ((2021,1),(2022,10)):
        target_k=event_to_k[(season,round_)]
        race=truth[truth.season.eq(season)&truth['round'].eq(round_)]
        for col in ('med_gap','matched_gap'):
            scores=[]
            for driver,code in zip(race.driver,race.code):
                numerator=denominator=0.
                for obs in source[source.code.eq(code)].itertuples():
                    k=event_to_k[(obs.season,obs.round)]
                    v=getattr(obs,col)
                    if k<target_k and pd.notna(v):
                        weight=.5**((target_k-k)/10)
                        numerator+=weight*v; denominator+=weight
                scores.append((numerator/denominator if denominator else np.inf,driver))
            expected=[driver for _,driver in sorted(scores)[:3]]
            saved=r[r.season.eq(season)&r['round'].eq(round_)&r.model.eq('PRIOR_'+col)].iloc[0]
            assert expected==[saved.pred1,saved.pred2,saved.pred3]


if __name__=='__main__':
    test_saved_audit()
    print('all E185 winner/overlap/team/coverage counts and scalar prior-source reconstruction passed')
