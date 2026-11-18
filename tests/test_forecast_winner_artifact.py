"""Independent saved-prediction metrics, paired resampling and feature artifact audit."""
import json
import numpy as np
import pandas as pd
from f1.data import ROOT,DATA
from experiments.e224_weather_winner import NEW,inputs,add_weather


def test_artifacts():
    R=pd.read_csv(ROOT/'results/e224_inner_per_race.csv',float_precision='round_trip')
    P=pd.read_csv(ROOT/'results/e224_inner_predictions.csv',float_precision='round_trip')
    summary=json.loads((ROOT/'results/e224_inner_summary.json').read_text())
    truth=pd.read_csv(DATA/'results.csv').loc[lambda d:d.season.between(2020,2022)]
    assert P.season.between(2020,2022).all() and R.groupby('model').size().eq(61).all()
    assert not P.duplicated(['season','round','driver','model']).any()
    assert P.groupby('model').size().eq(1220).all()
    measured=[]; winners={}; probability=[]
    for (season,rnd,name),g in P.groupby(['season','round','model'],sort=True):
        actual=truth[truth.season.eq(season)&truth['round'].eq(rnd)].sort_values('order')
        ranking=sorted(zip(g.score,g.driver))
        predicted=[driver for _,driver in ranking]
        winner=float(predicted[0]==actual.driver.iloc[0])
        podium=len(set(predicted[:3])&set(actual.driver.head(3)))/3
        measured.append((season,rnd,name,winner,podium))
        winners[(season,rnd,name)]=predicted[0]
        metric=R[R.season.eq(season)&R['round'].eq(rnd)&R.model.eq(name)].iloc[0]
        assert winner==metric.winner and abs(podium-metric.podium)<1e-14
        if name in ('weather_win','win_logit','weather_null'):
            weights=np.maximum(-g.score.to_numpy(),1e-12); weights=weights/weights.sum()
            y=g.driver.eq(actual.driver.iloc[0]).to_numpy(float)
            probability.append((season,rnd,name,-float(np.log(weights[y==1]).sum()),float(sum((weights-y)**2))))
    measured=pd.DataFrame(measured,columns=['season','round','model','winner','podium'])
    hits=measured.groupby('model').winner.sum().astype(int).to_dict()
    assert hits==summary['hits']
    a=measured[measured.model.eq('weather_win')].sort_values(['season','round']).winner.to_numpy()
    for ref in ('standings','ens','win_logit'):
        b=measured[measured.model.eq(ref)].sort_values(['season','round']).winner.to_numpy()
        rng=np.random.default_rng(0); indices=rng.integers(0,len(a),size=(5000,len(a)))
        resampled=(a[indices]-b[indices]).mean(axis=1)
        saved=summary['paired'][ref]
        assert abs(saved['difference']-float((a-b).mean()))<1e-14
        np.testing.assert_allclose(saved['ci'],np.percentile(resampled,[2.5,97.5]),rtol=0,atol=1e-14)
        assert saved['p_nonpositive']==float((resampled<=0).mean())
    pro=pd.DataFrame(probability,columns=['season','round','model','logloss','brier'])
    expected=pro.groupby('model')[['logloss','brier']].mean()
    for name in expected.index:
        for metric in ('logloss','brier'):
            assert abs(expected.loc[name,metric]-summary['probabilities'][name][metric])<1e-14
    gate=hits['weather_win']>=37 and all(v['ci'][0]>0 for v in summary['paired'].values()) and hits['weather_win']>hits['weather_null']
    assert gate==summary['gate']==False
    X,actual,forecasts,calendar=inputs(); X=add_weather(X,actual,forecasts,calendar)
    saved=pd.read_csv(ROOT/'results/e224_features.csv',float_precision='round_trip')
    np.testing.assert_allclose(saved[NEW],X[NEW],rtol=0,atol=1e-14)
    same=sum(winners[(s,r,'weather_win')]==winners[(s,r,'win_logit')]
             for s,r in measured[['season','round']].drop_duplicates().itertuples(index=False,name=None))
    print('Independent winner/strict-overlap/probability/bootstrap/gate/all2060-feature audit passes; identical winner picks',same,'/61')


if __name__=='__main__': test_artifacts()
