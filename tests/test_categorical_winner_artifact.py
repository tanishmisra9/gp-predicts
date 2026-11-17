"""E201 ranking/truth, probability quality, paired intervals and gate reconstruction."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def test_artifacts():
    P=pd.read_csv('results/e201_inner_predictions.csv',float_precision='round_trip')
    R=pd.read_csv('results/e201_inner_per_race.csv',float_precision='round_trip')
    C=pd.read_csv('results/e201_inner_probability_metrics.csv',float_precision='round_trip')
    result=json.loads(Path('results/e201_inner_summary.json').read_text())
    truth=pd.read_csv('data/results.csv').loc[lambda d:d.season.between(2020,2022),['season','round','driver','team','order']]
    assert P.season.max()==R.season.max()==C.season.max()==2022
    assert not P.duplicated(['season','round','driver','model']).any()
    assert P.groupby('model').size().eq(1220).all() and R.groupby('model').size().eq(61).all()
    verified=P.merge(truth,on=['season','round','driver'],suffixes=('_saved','_true'),validate='many_to_one')
    assert verified.team_saved.equals(verified.team_true) and verified.order_saved.equals(verified.order_true)
    np.testing.assert_array_equal(P.win,P.order.eq(1).astype(int))
    actual_team=truth[truth.order.eq(1)].set_index(['season','round']).team
    probability_models={'gbm_base','gbm_ids','ids_null','win_logit'}
    for name,g in P.groupby('model',sort=True):
        totals=[]; teams=0
        for (season,round_),event in g.groupby(['season','round'],sort=True):
            ordered=event.sort_values(['score','driver'])
            hit=int(ordered.order.iloc[0]==1)
            overlap=int(ordered.order.iloc[:3].le(3).sum())/3
            expected=R[R.model.eq(name)&R.season.eq(season)&R['round'].eq(round_)].iloc[0]
            assert hit==expected.winner and np.isclose(overlap,expected.podium)
            teams+=int(ordered.team.iloc[0]==actual_team.loc[(season,round_)])
            totals.append((season,hit,overlap))
            if name in probability_models:
                mass=np.maximum(-event.score.to_numpy(),1e-12)
                normalized=mass/sum(mass)
                np.testing.assert_allclose(event.p_win,normalized,atol=1e-12)
                assert np.isclose(event.p_win.sum(),1.)
                actual=event.order.eq(1).to_numpy(float)
                loss=-np.log(normalized[actual.astype(bool)]).item()
                brier=float(np.dot(normalized-actual,normalized-actual))
                stored=C[C.model.eq(name)&C.season.eq(season)&C['round'].eq(round_)].iloc[0]
                np.testing.assert_allclose([stored.logloss,stored.brier],[loss,brier],atol=1e-12)
            else:
                assert event.p_win.isna().all()
        summary=result['models'][name]; winners=sum(t[1] for t in totals)
        assert summary['winners']==winners and summary['team_hits']==teams
        np.testing.assert_allclose([summary['winner_rate'],summary['overlap']],[winners/61,sum(t[2] for t in totals)/61],atol=1e-12)
        by_season={str(s):sum(t[1] for t in totals if t[0]==s) for s in (2020,2021,2022)}
        assert summary['by_season']==by_season
        if name in probability_models:
            np.testing.assert_allclose([summary['logloss'],summary['brier']],C[C.model.eq(name)][['logloss','brier']].mean(),atol=1e-12)
    a=R[R.model.eq('gbm_ids')].set_index(['season','round']).sort_index().winner
    for name,reported in result['paired'].items():
        b=R[R.model.eq(name)].set_index(['season','round']).sort_index().winner
        d=(a-b).to_numpy()
        indexes=np.random.default_rng(0).integers(0,len(d),size=(5000,len(d)))
        draw=d[indexes].sum(axis=1)/len(d)
        np.testing.assert_allclose([reported['diff'],*reported['ci'],reported['p']],[d.sum()/len(d),*np.quantile(draw,[.025,.975]),(draw<=0).sum()/len(draw)],atol=1e-12)
    gate=result['models']['gbm_ids']['winners']>=37 and all(p['ci'][0]>0 for p in result['paired'].values()) and result['models']['gbm_ids']['winners']>result['models']['ids_null']['winners']
    assert result['gate']==gate
    old=pd.read_csv('results/e172_predictions.csv',float_precision='round_trip').loc[lambda d:d.season.le(2022)&d.model.eq('gbm_win')]
    matched=P[P.model.eq('gbm_base')].merge(old,on=['season','round','driver'],suffixes=('_new','_old'),validate='one_to_one')
    np.testing.assert_allclose(matched.score_new,matched.score_old,atol=1e-14,rtol=0)
    frozen=pd.read_csv('results/e181_inner_predictions.csv',float_precision='round_trip')
    frozen=frozen[frozen.model.isin(['standings','ens','win_logit'])]
    matched=P.merge(frozen,on=['season','round','driver','model'],suffixes=('_new','_old'),validate='one_to_one')
    np.testing.assert_array_equal(matched.score_new,matched.score_old)


if __name__=='__main__':
    test_artifacts()
    print('E201 official winners/overlap/teams, probabilities/losses, bootstrap/gate, numeric ablation and frozen references reproduce')
