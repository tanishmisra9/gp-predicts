"""E215: does historical matched-lap support calibrate teammate pace calls?"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import f1.walkforward as W
from experiments.a5_teammate_segments import PART
from f1.data import ROOT,load_results

BASE=['prior_abs_gap']; EXTRA=['prior_log_support']
KEYS=['season','round','team','code','mate']


def forecasts(targets,source):
    assert not targets.duplicated(KEYS).any() and not source.duplicated(KEYS).any()
    assert targets.season.between(2018,2022).all() and source.season.between(2018,2022).all()
    assert targets.code.lt(targets.mate).all() and source.code.lt(source.mate).all()
    assert np.isfinite(source[['gap','n']]).all().all() and source.n.ge(5).all()
    assert source.n.eq(source.n.astype(int)).all()
    legal=targets[KEYS]
    assert len(source.merge(legal,on=KEYS,validate='one_to_one'))==len(source)
    times=targets[['season','round','k']].drop_duplicates()
    assert not times.duplicated(['season','round']).any() and not times.k.duplicated().any()
    history=source.merge(times,on=['season','round'],validate='many_to_one')
    out=targets.copy(); out['prior_gap']=0.; out['support']=0.; out['history_events']=0
    for r in out.itertuples():
        past=history[history.k.lt(r.k)&history.team.eq(r.team)&history.code.eq(r.code)&history.mate.eq(r.mate)]
        w=past.n*.5**((r.k-past.k)/10)
        support=float(w.sum())
        out.loc[r.Index,'support']=support
        out.loc[r.Index,'history_events']=len(past)
        if support>0:
            out.loc[r.Index,'prior_gap']=float((w*past.gap).sum()/support)
    out['prior_abs_gap']=out.prior_gap.abs()
    out['prior_log_support']=np.log1p(out.support)
    return out


def probability(train,current,cols,shuffled=False):
    W.assert_pre_weekend(cols)
    assert train.empty or train.k.max()<current.k.min()
    y=train.correct.to_numpy().copy()
    if shuffled:
        rng=np.random.default_rng(215)
        for ids in train.groupby('k',sort=True).indices.values():
            y[ids]=rng.permutation(y[ids])
    if len(train)<50 or len(np.unique(y))<2:
        return np.full(len(current),(float(y.sum())+1)/(len(y)+2))
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=500))
    model.fit(train[cols],y,logisticregression__sample_weight=W._w(train,100))
    return model.predict_proba(current[cols])[:,1]


def call_labels(frame):
    correct=(frame.prior_gap.le(0)==frame.observed_gap.lt(0)).astype(float)
    correct.loc[frame.observed_gap.isna()|frame.observed_gap.eq(0)|frame.history_events.eq(0)]=np.nan
    return correct


def main():
    results=load_results().loc[lambda d:d.season.between(2018,2022)]
    times=results[['season','round']].drop_duplicates().sort_values(['season','round']).reset_index(drop=True)
    times['k']=np.arange(len(times))
    pairs=[]
    for (season,round_,team),g in results.groupby(['season','round','team'],sort=True):
        codes=sorted(g.code.unique()); assert len(codes)==2
        pairs.append(dict(season=season,round=round_,team=team,code=codes[0],mate=codes[1]))
    targets=pd.DataFrame(pairs).merge(times,on=['season','round'],validate='many_to_one')
    source=pd.read_csv(PART).loc[lambda d:d.season.between(2018,2022)&d.code.lt(d.mate)].copy()
    X=forecasts(targets,source)
    X=X.merge(source[KEYS+['gap']].rename(columns={'gap':'observed_gap'}),on=KEYS,how='left',validate='one_to_one')
    X['correct']=call_labels(X)
    X.to_csv(ROOT/'results/e215_pair_forecasts.csv',index=False)
    scored=[]
    for k,current in X[X.season.ge(2020)].groupby('k',sort=True):
        current=current[current.correct.notna()].copy()
        if current.empty:
            continue
        train=X[X.k.lt(k)&X.season.ge(2019)&X.correct.notna()]
        for name,cols,shuffle in [('mean_only',BASE,False),('support',BASE+EXTRA,False),('support_null',BASE+EXTRA,True)]:
            p=probability(train,current,cols,shuffled=shuffle)
            out=current[KEYS+['k','correct']].copy(); out['model']=name; out['probability']=p
            out['brier']=(p-current.correct.to_numpy())**2
            q=np.clip(p,1e-12,1-1e-12); y=current.correct.to_numpy()
            out['logloss']=-(y*np.log(q)+(1-y)*np.log1p(-q))
            scored.append(out)
    P=pd.concat(scored,ignore_index=True)
    race=P.groupby(['season','round','model'])[['brier','logloss']].mean().reset_index()
    comparisons={}
    for ref in ('mean_only','support_null'):
        comparisons[ref]={}
        for metric in ('brier','logloss'):
            wide=race.pivot(index=['season','round'],columns='model',values=metric)
            gain,ci,p=W.paired_boot(wide[ref],wide.support)
            comparisons[ref][metric]=dict(gain=float(gain),ci=ci.tolist(),p_nonpositive=float(p))
    eligible=X[X.season.ge(2020)&X.correct.notna()]
    coverage=int(eligible.groupby('k').size().ge(5).sum())
    base=comparisons['mean_only']; null=comparisons['support_null']
    gate=bool(coverage>=50 and len(eligible)>=300 and base['brier']['gain']>=.002 and
        all(base[m]['ci'][0]>0 and null[m]['ci'][0]>0 for m in ('brier','logloss')))
    summary=dict(pair_rows=len(eligible),races=int(eligible.k.nunique()),races_with_five_pairs=coverage,
        missing_inner_pairs=int(610-len(eligible)),paired=comparisons,gate=gate)
    P.to_csv(ROOT/'results/e215_inner_probabilities.csv',index=False)
    race.to_csv(ROOT/'results/e215_inner_race_metrics.csv',index=False)
    (ROOT/'results/e215_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('RESULT',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
