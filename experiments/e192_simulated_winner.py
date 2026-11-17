"""E192: same race draws, exact-winner versus podium probability selection."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import f1.walkforward as W
from experiments.e168_winner import win_score
from f1.data import DATA,load_results
from f1.features import build_features,pre_weekend_cols

DNF_F=['d10_dnf','t10_dnf','d3_dnf','t3_dnf','d_coll','c_dnf','era_round']


def add_reliability(X,results):
    keys=['season','round','code']
    source=results[keys+['circuit','status']].merge(X[keys+['k','era']],on=keys,validate='one_to_one')
    source['coll']=source.status.isin(['Collision','Accident','Collision damage','Spun off']).astype(float)
    source['dnf_y']=(~(source.status.isin(['Finished','Lapped'])|source.status.str.startswith('+'))&~source.status.isin(['Did not start','Withdrew'])).astype(float)
    circuit=source.groupby(['k','circuit']).dnf_y.mean().reset_index()
    out=X.merge(source[keys+['dnf_y']],on=keys,validate='one_to_one')
    out['d_coll']=np.nan; out['c_dnf']=np.nan
    for k,cur in out.groupby('k',sort=True):
        past=source[source.k.lt(k)]
        weights=.5**((k-past.k)/6)*np.where(past.era<cur.era.iloc[0],.5,1.)
        collisions=(past.coll*weights).groupby(past.code).sum()/weights.groupby(past.code).sum()
        out.loc[cur.index,'d_coll']=cur.code.map(collisions)
        name=source[source.k.eq(k)].circuit.iloc[0]
        out.loc[cur.index,'c_dnf']=circuit[circuit.k.lt(k)&circuit.circuit.eq(name)].dnf_y.mean()
    return out


def probabilities_from_performance(performance):
    assert performance.ndim==2 and performance.shape[1]>0 and not np.isnan(performance).any()
    valid=performance[np.isfinite(performance).any(axis=1)]
    assert len(valid)>0,'no simulated survivors'
    ranking=np.argsort(valid,axis=1,kind='stable')
    wins=np.bincount(ranking[:,0],minlength=valid.shape[1])
    podium=np.bincount(ranking[:,:min(3,valid.shape[1])].ravel(),minlength=valid.shape[1])
    return wins/len(valid),podium/len(valid),(wins+.5)/(len(valid)+.5*valid.shape[1]),len(valid)


def simulate(tr,te,F,shuffled=False,reliability_features=None):
    risk_features=DNF_F if reliability_features is None else reliability_features
    W.assert_pre_weekend(F+risk_features)
    assert tr.k.max()<te.k.min()
    train=tr.sort_values(['k','driver']).copy(); current=te.sort_values('driver')
    if shuffled:
        rng=np.random.default_rng(192)
        for _,ids in train.groupby('k',sort=True).groups.items():
            train.loc[ids,['y_pace','dnf_y']]=train.loc[rng.permutation(ids),['y_pace','dnf_y']].to_numpy()
    pace=train[train.y_pace.notna()]
    model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=500))
    model.fit(pace[F],pace.y_pace,ridge__sample_weight=W._w(pace,100))
    mean=model.predict(current[F])
    retirement=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=.1,max_iter=500))
    retirement.fit(train[risk_features],train.dnf_y,logisticregression__sample_weight=W._w(train,100))
    risk=retirement.predict_proba(current[risk_features])[:,1]
    rng=np.random.default_rng(192000+int(current.season.iloc[0])*100+int(current['round'].iloc[0]))
    performance=mean+.6*np.std(mean)*rng.standard_normal((4000,len(current)))
    performance[rng.random(performance.shape)<risk]=np.inf
    win,podium,smooth,count=probabilities_from_performance(performance)
    return current[['season','round','driver']].assign(win_probability=win,podium_probability=podium,smooth_win=smooth,effective_draws=count)


def main(reliability_builder=add_reliability,reliability_features=DNF_F,experiment=192):
    results=load_results().loc[lambda d:d.season.le(2022)]
    pace=pd.read_csv(DATA/'racepace_ff1.csv').loc[lambda d:d.season.le(2022)]
    X=build_features(results,pace); F=pre_weekend_cols(X)
    X=reliability_builder(X,results).merge(pace[['season','round','code','pace_gap']].rename(columns={'pace_gap':'y_pace'}),on=['season','round','code'],how='left',validate='one_to_one')
    rows=[]
    for _,cur in X[X.season.between(2020,2022)].groupby('k',sort=True):
        tr=X[X.k.lt(cur.k.iloc[0])&X.season.ge(2019)]
        for null in (False,True):
            rows.append(simulate(tr,cur,F,shuffled=null,reliability_features=reliability_features).assign(null=null))
    simulation=pd.concat(rows,ignore_index=True)
    normal=simulation[~simulation['null']].rename(columns={'win_probability':'prior_mc_win','podium_probability':'prior_mc_podium'})
    randomized=simulation[simulation['null']].rename(columns={'win_probability':'prior_mc_null'})
    keys=['season','round','driver']
    X=X.merge(normal[keys+['prior_mc_win','prior_mc_podium']],on=keys,how='left',validate='one_to_one')
    X=X.merge(randomized[keys+['prior_mc_null']],on=keys,how='left',validate='one_to_one')
    W.MODELS.update(
        standings=lambda tr,te,F:(-te.d_season_pts.fillna(0)*100+te.d10_order.fillna(11)*1e-3).to_numpy(),
        pre_logit=lambda tr,te,F:W.m_logit_pod(tr,te,F,C=.03,hl=100),
        win_logit=win_score,win_null=lambda tr,te,F:win_score(tr,te,F,shuffled=True),
        mc_win=lambda tr,te,F:-te.prior_mc_win.to_numpy(),
        mc_podium=lambda tr,te,F:-te.prior_mc_podium.to_numpy(),
        mc_null=lambda tr,te,F:-te.prior_mc_null.to_numpy())
    R,P=W.walk_forward(X,[2020,2021,2022],['standings','pre_logit','win_logit','win_null','mc_win','mc_podium','mc_null'],F=F,ens={'ens':['standings','pre_logit']},tie=None)
    label=results[['season','round','driver','team','order']]
    chosen=P.sort_values(['season','round','model','score','driver']).drop_duplicates(['season','round','model']).merge(label,on=keys,validate='many_to_one')
    truth=label[label.order.eq(1)][['season','round','team']].rename(columns={'team':'winner_team'})
    chosen=chosen.merge(truth,on=['season','round'],validate='many_to_one')
    team=chosen.assign(hit=chosen.team.eq(chosen.winner_team)).groupby('model').hit.sum()
    print(R.groupby('model')[['winner','podium']].agg(['sum','mean']).round(4).to_string(),flush=True)
    print('winning team',team.to_dict(),flush=True)
    print(R.pivot_table(index='season',columns='model',values='winner',aggfunc='sum').to_string(),flush=True)
    pivot=R.pivot(index=['season','round'],columns='model',values='winner'); intervals={}
    for ref in ('standings','ens','win_logit','mc_podium'):
        delta,ci,p=W.paired_boot(pivot.mc_win,pivot[ref]); intervals[ref]=(delta,ci)
        print('mc_win minus',ref,delta,ci,p,flush=True)
    probs=simulation.merge(label,on=keys,validate='many_to_one')
    print('simulated winner logloss',probs[probs.order.eq(1)].assign(loss=lambda d:-np.log(d.smooth_win)).groupby('null').loss.mean().to_dict(),flush=True)
    gate=pivot.mc_win.sum()>=35 and team.mc_win>=team.win_logit and all(intervals[ref][0]>0 and intervals[ref][1][0]>0 for ref in ('win_logit','mc_podium')) and pivot.mc_win.sum()>pivot.mc_null.sum()
    print('INNER GATE',bool(gate),flush=True)
    assert R.groupby('model').size().eq(61).all() and R.season.max()==2022
    R.to_csv(f'results/e{experiment}_inner_per_race.csv',index=False); P.to_csv(f'results/e{experiment}_inner_predictions.csv',index=False)
    simulation.to_csv(f'results/e{experiment}_inner_simulation_probabilities.csv',index=False)


if __name__=='__main__':
    main()
