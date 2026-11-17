"""E186: retrospective driver conversion measurement, never a pre-weekend model."""
import numpy as np
import pandas as pd

import f1.walkforward as W
from f1.data import DATA, load_results



def historical_residual(state, driver, k):
    if driver not in state:
        return 0.
    last, numerator, denominator = state[driver]
    w = .5**((k-last)/10)
    return numerator*w/(20+denominator*w)


def record_residual(state, driver, k, value):
    last, numerator, denominator = state.get(driver,(k,0.,0.))
    w = .5**((k-last)/10)
    state[driver] = (k,numerator*w+value,denominator*w+1)


def corrected_probability(base, residual):
    p=np.clip(base+residual,1e-6,1.)
    return p/p.sum()


def conversion_trace(results, pace):
    assert results.season.le(2022).all() and pace.season.le(2022).all(), 'E186 is inner-only'
    keys=['season','round','driver']
    assert not results.duplicated(keys).any()
    d=results.merge(pace[['season','round','code','pace_gap']],on=['season','round','code'],how='left',validate='one_to_one')
    bucket_count=np.zeros(6); bucket_wins=np.zeros(6)
    state={}; null_state={}; rng=np.random.default_rng(186); rows=[]
    for k,((season,round_),g) in enumerate(d.groupby(['season','round'],sort=True)):
        g=g.sort_values('driver').copy().reset_index(drop=True)
        assert sorted(g.order)==list(range(1,len(g)+1))
        prior=np.array([historical_residual(state,driver,k) for driver in g.driver])
        null_prior=np.array([historical_residual(null_state,driver,k) for driver in g.driver])
        ranked=g.sort_values(['pace_gap','driver'],na_position='last').index
        rank=np.empty(len(g),dtype=int); rank[ranked]=np.arange(1,len(g)+1)
        bucket=np.select([rank==1,rank==2,rank==3,rank<=6,rank<=10],[0,1,2,3,4],default=5)
        bucket[g.pace_gap.isna().to_numpy()]=5
        expected=(bucket_wins[bucket]+1)/(bucket_count[bucket]+2)
        base=expected/expected.sum()
        corrected=corrected_probability(base,prior)
        null=corrected_probability(base,null_prior)
        residual=g.order.eq(1).to_numpy(dtype=float)-base
        eligible=(rank<=3)&g.pace_gap.notna().to_numpy()
        rows.append(g.assign(k=k,prior_driver_residual=prior,prior_null_residual=null_prior,
            base_probability=base,corrected_probability=corrected,null_probability=null,
            residual=residual,eligible=eligible,pace_rank=rank))
        ids=np.flatnonzero(eligible)
        permutation=rng.permutation(ids)
        for i,j in zip(ids,permutation):
            record_residual(state,g.driver.iloc[i],k,residual[i])
            record_residual(null_state,g.driver.iloc[i],k,residual[j])
        np.add.at(bucket_count,bucket,1)
        np.add.at(bucket_wins,bucket,g.order.eq(1).astype(float))
    return pd.concat(rows,ignore_index=True)


def residual_correlation(trace):
    eligible=trace[trace.eligible]
    def corr(frame):
        a=frame.prior_driver_residual.to_numpy(); b=frame.residual.to_numpy()
        return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else np.nan
    groups=[g for _,g in eligible.groupby(['season','round'],sort=True)]
    rng=np.random.default_rng(186)
    samples=[corr(pd.concat([groups[i] for i in rng.integers(len(groups),size=len(groups))])) for _ in range(4000)]
    return corr(eligible),np.nanpercentile(samples,[2.5,97.5]),len(eligible)


def main():
    df=load_results().loc[lambda d:d.season.le(2022)]
    pace=pd.read_csv(DATA/'racepace_ff1.csv').loc[lambda d:d.season.le(2022)]
    trace=conversion_trace(df,pace)
    inner=trace[trace.season.ge(2020)].copy()
    rows=[]
    for (season,round_),g in inner.groupby(['season','round'],sort=True):
        for name,col in (('HINDSIGHT_pace_base','base_probability'),('HINDSIGHT_driver_conversion','corrected_probability'),('HINDSIGHT_null_history','null_probability')):
            probability=g[col].to_numpy()
            assert np.isfinite(probability).all() and np.isclose(probability.sum(),1.)
            metrics=W.metrics(g,-probability,tie=None)
            chosen=g.assign(probability=probability).sort_values(['probability','driver'],ascending=[False,True]).iloc[:3]
            actual=g[g.order.eq(1)].iloc[0]
            rows.append(dict(season=season,round=round_,model=name,**metrics,
                logloss=-np.log(g.loc[g.order.eq(1),col].iloc[0]),
                team_hit=float(chosen.iloc[0].team==actual.team),
                pred1=chosen.iloc[0].driver,pred2=chosen.iloc[1].driver,pred3=chosen.iloc[2].driver))
    R=pd.DataFrame(rows)
    assert R.groupby('model').size().eq(61).all() and R.season.max()==2022
    corr,ci,n=residual_correlation(inner)
    print('eligible residual correlation',corr,'race-bootstrap CI',ci,'entries',n,flush=True)
    print(R.groupby('model')[['winner','podium','team_hit','logloss']].agg(['sum','mean']).round(5).to_string(),flush=True)
    pivot=R.pivot(index=['season','round'],columns='model',values='winner')
    delta,winner_ci,_=W.paired_boot(pivot.HINDSIGHT_driver_conversion,pivot.HINDSIGHT_pace_base)
    loss=R.pivot(index=['season','round'],columns='model',values='logloss')
    improvement,loss_ci,_=W.paired_boot(loss.HINDSIGHT_pace_base,loss.HINDSIGHT_driver_conversion)
    print('winner difference',delta,winner_ci,'logloss improvement',improvement,loss_ci,flush=True)
    gate=corr>=.15 and pivot.HINDSIGHT_driver_conversion.sum()>=46 and delta>0 and winner_ci[0]>0 and improvement>0 and loss_ci[0]>0 and pivot.HINDSIGHT_driver_conversion.sum()>pivot.HINDSIGHT_null_history.sum()
    print('MEASUREMENT GATE',bool(gate),flush=True)
    print(R.pivot_table(index='season',columns='model',values='winner',aggfunc='sum').to_string(),flush=True)
    R.to_csv('results/e186_inner_hindsight_metrics.csv',index=False)
    inner.to_csv('results/e186_inner_conversion_trace.csv',index=False)


if __name__=='__main__':
    main()
