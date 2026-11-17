"""E201 fixed native categorical identities; inner-only winner experiment."""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.e172_nonlinear_winner import winner_gbm
from f1.data import DATA,ROOT,load_results
from f1.features import build_features,pre_weekend_cols

CATS=['driver','team','circuit_id']


def attach_context(X,results):
    events=results[['season','round','circuit']].drop_duplicates()
    assert not events.duplicated(['season','round']).any() and events.circuit.notna().all()
    out=X.merge(events.rename(columns={'circuit':'circuit_id'}),on=['season','round'],how='left',validate='many_to_one')
    assert out.circuit_id.notna().all()
    return out


def category_inputs(train,current,F):
    W.assert_pre_weekend(F+CATS)
    assert len(train)>0 and train.k.max()<current.k.min()
    a=train[F].copy(); b=current[F].copy()
    for name in CATS:
        prior=train[name].fillna('__missing__').astype(str)
        vocabulary=sorted(prior.unique())
        a[name]=pd.Categorical(prior,categories=vocabulary)
        b[name]=pd.Categorical(current[name].fillna('__missing__').astype(str),categories=vocabulary)
    return a,b


def labels(train,shuffled):
    y=train.win.to_numpy().copy()
    if shuffled:
        y[:]=0; rng=np.random.default_rng(201)
        for indices in train.groupby('k',sort=True).indices.values():
            y[rng.choice(indices)]=1
        assert y.sum()==train.k.nunique()
    return y


def categorical_score(train,current,F,shuffled=False):
    a,b=category_inputs(train,current,F)
    model=lgb.LGBMClassifier(**W._lgb_params(min_data_per_group=30))
    model.fit(a,labels(train,shuffled),sample_weight=W._w(train,100),categorical_feature=CATS)
    return -model.predict_proba(b)[:,1]


def main():
    results=load_results().loc[lambda d:d.season.le(2022)]
    pace=pd.read_csv(DATA/'racepace_ff1.csv').loc[lambda d:d.season.le(2022)]
    X=attach_context(build_features(results,pace),results); F=pre_weekend_cols(X)
    W.assert_pre_weekend(F+CATS)
    W.MODELS.update(gbm_base=winner_gbm,gbm_ids=categorical_score,ids_null=lambda tr,te,F:categorical_score(tr,te,F,shuffled=True))
    R,P=W.walk_forward(X,[2020,2021,2022],['gbm_base','gbm_ids','ids_null'],F=F,tie=None)
    references=['standings','ens','win_logit']
    old_R=pd.read_csv(ROOT/'results/e181_inner_per_race.csv')
    old_P=pd.read_csv(ROOT/'results/e181_inner_predictions.csv',float_precision='round_trip')
    R=pd.concat([R,old_R[old_R.model.isin(references)]],ignore_index=True)
    P=pd.concat([P,old_P[old_P.model.isin(references)]],ignore_index=True)
    assert R.groupby('model').size().eq(61).all() and P.groupby('model').size().eq(1220).all()
    old_base=pd.read_csv(ROOT/'results/e172_predictions.csv',float_precision='round_trip').loc[lambda d:d.season.le(2022)&d.model.eq('gbm_win')]
    check=P[P.model.eq('gbm_base')].merge(old_base,on=['season','round','driver'],suffixes=('_new','_old'),validate='one_to_one')
    np.testing.assert_allclose(check.score_new,check.score_old,atol=1e-14,rtol=0)
    truth=X[['season','round','driver','team','win','order']]
    scored=P.merge(truth,on=['season','round','driver'],validate='many_to_one')
    scored['p_win']=np.nan; probabilistic=['gbm_base','gbm_ids','ids_null','win_logit']
    eligible=scored.model.isin(probabilistic)
    positive=(-scored.loc[eligible,'score']).clip(lower=1e-12)
    total=positive.groupby([scored.loc[eligible,c] for c in ('season','round','model')]).transform('sum')
    scored.loc[eligible,'p_win']=positive/total
    calibration=[]
    for (season,round_,name),g in scored[eligible].groupby(['season','round','model'],sort=True):
        calibration.append(dict(season=int(season),round=int(round_),model=name,logloss=float(-np.log(g.loc[g.win.eq(1),'p_win']).sum()),brier=float(((g.p_win-g.win)**2).sum())))
    calibration=pd.DataFrame(calibration)
    winner_team=X[X.win.eq(1)].set_index(['season','round']).team
    picked=scored.sort_values(['score','driver']).drop_duplicates(['season','round','model'])
    picked['team_hit']=[team==winner_team.loc[(s,r)] for s,r,team in picked[['season','round','team']].itertuples(index=False,name=None)]
    summary={}
    for name,g in R.groupby('model',sort=True):
        summary[name]=dict(winners=int(g.winner.sum()),winner_rate=float(g.winner.mean()),overlap=float(g.podium.mean()),team_hits=int(picked[picked.model.eq(name)].team_hit.sum()),by_season=g.groupby('season').winner.sum().astype(int).to_dict())
        if name in probabilistic:
            summary[name].update(calibration[calibration.model.eq(name)][['logloss','brier']].mean().to_dict())
    a=R[R.model.eq('gbm_ids')].set_index(['season','round']).sort_index().winner
    paired={}
    for name in ['gbm_base']+references:
        b=R[R.model.eq(name)].set_index(['season','round']).sort_index().winner
        diff,ci,p=W.paired_boot(a,b)
        paired[name]=dict(diff=float(diff),ci=ci.tolist(),p=float(p))
    gate=summary['gbm_ids']['winners']>=37 and all(value['ci'][0]>0 for value in paired.values()) and summary['gbm_ids']['winners']>summary['ids_null']['winners']
    result=dict(models=summary,paired=paired,gate=bool(gate))
    (ROOT/'results/e201_inner_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    R.to_csv(ROOT/'results/e201_inner_per_race.csv',index=False)
    scored.to_csv(ROOT/'results/e201_inner_predictions.csv',index=False)
    calibration.to_csv(ROOT/'results/e201_inner_probability_metrics.csv',index=False)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
