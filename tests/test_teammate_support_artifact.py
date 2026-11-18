"""Independent E215 scalar history, prefix fits, proper scores and race bootstrap."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from experiments.a5_teammate_segments import PART,prior_pair
from experiments.e215_teammate_support import BASE,EXTRA,KEYS,forecasts,probability
from f1.data import load_results


def test_artifacts():
    X=pd.read_csv('results/e215_pair_forecasts.csv',float_precision='round_trip')
    P=pd.read_csv('results/e215_inner_probabilities.csv',float_precision='round_trip')
    R=pd.read_csv('results/e215_inner_race_metrics.csv',float_precision='round_trip')
    summary=json.loads(Path('results/e215_summary.json').read_text())
    source=pd.read_csv(PART).loc[lambda d:d.season.between(2018,2022)&d.code.lt(d.mate)]
    assert len(X)==1030 and X.season.max()==2022 and X.k.nunique()==103
    assert not X.duplicated(KEYS).any()
    official=load_results().loc[lambda d:d.season.between(2018,2022)]
    events=sorted(set(zip(official.season,official['round'])))
    expected=[]
    for (season,round_,team),g in official.groupby(['season','round','team'],sort=True):
        codes=sorted(g.code.unique()); assert len(codes)==2
        expected.append((season,round_,team,codes[0],codes[1],events.index((season,round_))))
    assert list(X[KEYS+['k']].itertuples(index=False,name=None))==expected
    metadata={(r.season,r.round):r.k for r in X[['season','round','k']].drop_duplicates().itertuples()}
    rows=[(metadata[(r.season,r.round)],r.team,r.code,r.mate,r.gap,r.n) for r in source.itertuples()]
    actual={(r.season,r.round,r.team,r.code,r.mate):r.gap for r in source.itertuples()}
    for r in X.itertuples():
        a=0.; b=0.; count=0
        for k,team,code,mate,gap,n in rows:
            if k<r.k and (team,code,mate)==(r.team,r.code,r.mate):
                w=n*2**(-(r.k-k)/10); a+=w*gap; b+=w; count+=1
        np.testing.assert_allclose(r.prior_gap,a/b if b else 0.,atol=1e-13)
        np.testing.assert_allclose(r.support,b,atol=1e-12)
        assert r.history_events==count
        target=actual.get((r.season,r.round,r.team,r.code,r.mate),np.nan)
        np.testing.assert_allclose(r.observed_gap,target,atol=1e-15,equal_nan=True)
        expected=np.nan if count==0 or not np.isfinite(target) or target==0 else float((r.prior_gap<=0)==(target<0))
        np.testing.assert_allclose(r.correct,expected,equal_nan=True)
    np.testing.assert_allclose(X.prior_abs_gap,abs(X.prior_gap),atol=1e-15)
    np.testing.assert_allclose(X.prior_log_support,np.log1p(X.support),atol=1e-15)
    reference=prior_pair(X[KEYS+['k']],source)
    check=X.merge(reference,on=KEYS,validate='one_to_one',suffixes=('','_original'))
    np.testing.assert_allclose(check.prior_gap,check.prior_gap_original.fillna(0),atol=1e-13)
    eligible=X[X.season.ge(2020)&X.correct.notna()]
    assert len(P)==len(eligible)*3 and P['round'].notna().all()
    for k,current in eligible.groupby('k',sort=True):
        train=X[X.k.lt(k)&X.season.ge(2019)&X.correct.notna()]
        for name,cols,shuffle in [('mean_only',BASE,False),('support',BASE+EXTRA,False),('support_null',BASE+EXTRA,True)]:
            y=train.correct.to_numpy().copy()
            if shuffle:
                rng=np.random.default_rng(215)
                for ids in train.groupby('k',sort=True).indices.values():
                    y[ids]=rng.permutation(y[ids])
            if len(y)<50 or len(np.unique(y))<2:
                expected=np.full(len(current),(sum(y)+1)/(len(y)+2))
            else:
                a=train[cols].to_numpy(); b=current[cols].to_numpy()
                mean=a.mean(axis=0); scale=a.std(axis=0); scale[scale==0]=1
                model=LogisticRegression(C=.03,max_iter=500)
                model.fit((a-mean)/scale,y,sample_weight=2**(-(train.k.max()-train.k.to_numpy())/100))
                expected=model.predict_proba((b-mean)/scale)[:,1]
            saved=P[P.k.eq(k)&P.model.eq(name)]
            np.testing.assert_array_equal(saved[KEYS],current[KEYS])
            np.testing.assert_allclose(saved.probability,expected,atol=1e-12,rtol=1e-12)
            np.testing.assert_array_equal(saved.correct,current.correct)
    assert P.probability.between(0,1,inclusive='neither').all()
    np.testing.assert_allclose(P.brier,(P.probability-P.correct)**2,atol=1e-15)
    q=P.probability.clip(1e-12,1-1e-12)
    np.testing.assert_allclose(P.logloss,-(P.correct*np.log(q)+(1-P.correct)*np.log1p(-q)),atol=1e-15)
    race=P.groupby(['season','round','model'])[['brier','logloss']].mean().reset_index()
    pd.testing.assert_frame_equal(R,race,atol=1e-14,rtol=1e-14)
    for ref,comparisons in summary['paired'].items():
        for metric,comparison in comparisons.items():
            wide=race.pivot(index=['season','round'],columns='model',values=metric)
            d=(wide[ref]-wide.support).to_numpy()
            ids=np.random.default_rng(0).integers(0,len(d),size=(5000,len(d)))
            sample=d[ids].mean(axis=1)
            np.testing.assert_allclose(comparison['gain'],d.mean(),atol=1e-15)
            np.testing.assert_allclose(comparison['ci'],np.quantile(sample,[.025,.975]),atol=1e-15)
            np.testing.assert_allclose(comparison['p_nonpositive'],(sample<=0).mean(),atol=1e-15)
    count=int(eligible.groupby('k').size().ge(5).sum())
    assert summary['pair_rows']==len(eligible) and summary['races']==eligible.k.nunique()
    assert summary['missing_inner_pairs']==610-len(eligible) and summary['races_with_five_pairs']==count
    b=summary['paired']['mean_only']; n=summary['paired']['support_null']
    assert summary['gate']==bool(count>=50 and len(eligible)>=300 and b['brier']['gain']>=.002 and all(b[m]['ci'][0]>0 and n[m]['ci'][0]>0 for m in ('brier','logloss')))
    targets=X[KEYS+['k']]; cutoff=metadata[(2022,10)]
    changed=source.copy(); future=[metadata[(r.season,r.round)]>=cutoff for r in source.itertuples()]
    changed.loc[future,['gap','n']]=[99.,1000]
    a=forecasts(targets,source); b=forecasts(targets,changed)
    pd.testing.assert_frame_equal(a[a.k.le(cutoff)],b[b.k.le(cutoff)])
    prefix=forecasts(targets,source[np.logical_not(future)])
    pd.testing.assert_frame_equal(a[a.k.eq(cutoff)],prefix[prefix.k.eq(cutoff)])
    train=X[X.k.lt(cutoff)&X.season.ge(2019)&X.correct.notna()]
    current=X[X.k.eq(cutoff)]
    np.testing.assert_array_equal(probability(train,current,BASE+EXTRA),probability(train,b[b.k.eq(cutoff)].assign(correct=999,gridn=-999),BASE+EXTRA))
    earlier=source.copy(); earlier.loc[earlier.season.eq(2021),'n']*=2
    assert not forecasts(targets,earlier).loc[a.k.eq(cutoff),'support'].equals(a.loc[a.k.eq(cutoff),'support'])
    print('MEAN_SCORES',R.groupby('model')[['brier','logloss']].mean().to_json())


if __name__=='__main__':
    test_artifacts()
    print('E215 independent scalar means/support/E108 parity, all prefix fits, labels, scores, bootstrap and timing PASS')
