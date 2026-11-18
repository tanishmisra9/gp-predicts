"""E214 independent rankings, probabilities, references and race bootstrap."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from f1.data import load_results


def test_scores():
    P=pd.read_csv('results/e214_inner_predictions.csv',float_precision='round_trip')
    R=pd.read_csv('results/e214_inner_per_race.csv',float_precision='round_trip')
    result=json.loads(Path('results/e214_inner_summary.json').read_text())
    truth=load_results().loc[lambda d:d.season.between(2020,2022),['season','round','driver','order']]
    joined=P[['season','round','driver','model','score']].merge(truth,on=['season','round','driver'],validate='many_to_one')
    assert len(joined)==len(P)==61*20*6 and np.isfinite(joined.score).all()
    joined=joined.sort_values(['season','round','model','score','driver'])
    joined['rank']=joined.groupby(['season','round','model']).cumcount()+1
    joined['winner_hit']=joined['rank'].eq(1)&joined.order.eq(1)
    joined['podium_hit']=joined['rank'].le(3)&joined.order.le(3)
    actual=joined.groupby(['season','round','model'])[['winner_hit','podium_hit']].sum()
    saved=R.set_index(['season','round','model']).sort_index()
    np.testing.assert_array_equal(actual.winner_hit,saved.winner)
    np.testing.assert_allclose(actual.podium_hit/3,saved.podium,atol=1e-14)
    hits=actual.groupby('model').winner_hit.sum().astype(int).to_dict()
    assert hits==result['hits']
    for name,value in result['overlap'].items():
        np.testing.assert_allclose(value,actual.xs(name,level='model').podium_hit.mean()/3,atol=1e-15)
    wide=actual.winner_hit.unstack('model')
    for name,comparison in result['paired'].items():
        d=(wide.consistency_win-wide[name]).to_numpy(float)
        indices=np.random.default_rng(0).integers(0,len(d),size=(5000,len(d)))
        samples=d[indices].mean(axis=1)
        np.testing.assert_allclose(comparison['difference'],d.mean(),atol=1e-15)
        np.testing.assert_allclose(comparison['ci'],np.quantile(samples,[.025,.975]),atol=1e-15)
        np.testing.assert_allclose(comparison['p_nonpositive'],(samples<=0).mean(),atol=1e-15)
    assert result['gate']==bool(hits['consistency_win']>=37 and all(c['ci'][0]>0 for c in result['paired'].values()) and hits['consistency_win']>hits['consistency_null'])
    ref=pd.read_csv('results/e201_inner_predictions.csv',float_precision='round_trip')
    for name in ('standings','ens','win_logit'):
        a=P[P.model.eq(name)].sort_values(['season','round','driver'])
        b=ref[ref.model.eq(name)].sort_values(['season','round','driver'])
        np.testing.assert_array_equal(a[['season','round','driver']],b[['season','round','driver']])
        np.testing.assert_allclose(a.score,b.score,atol=1e-14,rtol=1e-14)
    probability=[]
    for name in ('consistency_win','win_logit','consistency_null'):
        for (season,round_),g in joined[joined.model.eq(name)].groupby(['season','round']):
            raw=-g.score.to_numpy(); assert (raw>0).all() and (raw<1).all()
            p=raw/raw.sum(); y=g.order.eq(1).to_numpy(float)
            probability.append(dict(model=name,season=int(season),round=int(round_),logloss=float(-np.log(p[y.astype(bool)][0])),brier=float(np.sum((p-y)**2))))
    metrics=pd.DataFrame(probability)
    for metric in ('logloss','brier'):
        w=metrics.pivot(index=['season','round'],columns='model',values=metric)
        d=(w.win_logit-w.consistency_win).to_numpy()
        indices=np.random.default_rng(0).integers(0,len(d),size=(5000,len(d)))
        print('PROBABILITY_GAIN',metric,float(d.mean()),np.quantile(d[indices].mean(axis=1),[.025,.975]).tolist(),flush=True)
    print('PROBABILITY',metrics.groupby('model')[['logloss','brier']].mean().to_json(),flush=True)
    print('SEASONAL',R.pivot_table(index='season',columns='model',values='winner',aggfunc='sum').to_json(),flush=True)


if __name__=='__main__':
    test_scores()
    print('E214 independent ranking/probability/reference/bootstrap/gate PASS')
