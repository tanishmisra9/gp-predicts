"""E214: ordinary within-stint lap spread, one prior-driver winner feature."""
import json

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.e168_winner import win_score
from experiments.e190_traffic_feasibility import cached_session,start_exposure
from f1.data import DATA,ROOT,load_results
from f1.features import build_features,pre_weekend_cols

NEW=['prior_driver_lap_spread']


def stint_spreads(laps):
    assert not laps.duplicated(['Driver','LapNumber']).any()
    clean=start_exposure(laps)
    rows=[]
    for (driver,stint,compound),g in clean.groupby(['Driver','Stint','Compound'],sort=True):
        if compound not in ('SOFT','MEDIUM','HARD') or len(g)<8:
            continue
        g=g.sort_values('LapNumber')
        x=g.LapNumber.to_numpy(float); y=g.seconds.to_numpy(float)
        assert np.isfinite(x).all() and np.isfinite(y).all() and (y>0).all()
        x=x-x.mean(); design=np.column_stack([np.ones(len(x)),x])
        coefficients,_,rank,_=np.linalg.lstsq(design,y,rcond=1e-10)
        assert rank==2
        residual=y-design@coefficients
        spread=100*1.4826*np.median(np.abs(residual-np.median(residual)))/np.median(y)
        rows.append(dict(code=driver,stint=stint,compound=compound,laps=len(g),spread=float(spread)))
    return pd.DataFrame(rows,columns=['code','stint','compound','laps','spread'])


def event_measurements(stints,entries):
    assert not stints.duplicated(['season','round','code','stint','compound']).any()
    assert np.isfinite(stints.spread).all() and stints.spread.ge(0).all()
    keys=['season','round','code']
    assert len(stints.merge(entries[keys],on=keys,validate='many_to_one'))==len(stints)
    values=stints.groupby(keys,sort=True).spread.median().rename('spread').reset_index()
    out=entries[keys].merge(values,on=keys,how='left',validate='one_to_one')
    field=out.groupby(['season','round']).spread
    out['centered_spread']=(out.spread-field.transform('mean')).where(field.transform('count').ge(10))
    return out


def add_history(X,source):
    keys=['season','round','code']
    assert not source.duplicated(keys).any()
    assert np.isfinite(source.centered_spread.dropna()).all()
    history=source.merge(X[keys+['k']],on=keys,validate='one_to_one')
    assert len(history)==len(source)
    out=X.copy(); out[NEW[0]]=0.
    for k,current in X.groupby('k',sort=True):
        past=history[history.k.lt(k)&history.centered_spread.notna()]
        w=.5**((k-past.k)/10)
        numer=(w*past.centered_spread).groupby(past.code).sum()
        denom=w.groupby(past.code).sum()
        out.loc[current.index,NEW[0]]=current.code.map(numer/(denom+5)).fillna(0.)
    return out


def extract():
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True)
    entries=load_results().loc[lambda d:d.season.between(2018,2022),['season','round','code']]
    parts=[]; reports=[]
    for season,round_ in entries[['season','round']].drop_duplicates().itertuples(index=False,name=None):
        report=dict(season=int(season),round=int(round_),available=False)
        try:
            session=cached_session(fastf1,int(season),int(round_))
            raw=pd.DataFrame(session.laps)
        except Exception as exc:
            report['error_type']=type(exc).__name__
        else:
            part=stint_spreads(raw).assign(season=season,round=round_)
            parts.append(part); report.update(available=True,stints=len(part))
        reports.append(report); print('SOURCE',json.dumps(report),flush=True)
    stints=pd.concat(parts,ignore_index=True)
    source=event_measurements(stints,entries)
    coverage=float(source.centered_spread.notna().mean())
    gate=bool(sum(r['available'] for r in reports)>=95 and coverage>=.8)
    stints.to_csv(ROOT/'results/e214_stint_spreads.csv',index=False)
    source.to_csv(ROOT/'results/e214_consistency_source.csv',index=False)
    (ROOT/'results/e214_source_summary.json').write_text(json.dumps(dict(events=reports,coverage=coverage,gate=gate),indent=2)+'\n')
    assert gate,'Registered source coverage gate failed; no model fit'
    return source


def main():
    source=extract()
    df=load_results().loc[lambda d:d.season.le(2022)]
    pace=pd.read_csv(DATA/'racepace_ff1.csv').loc[lambda d:d.season.le(2022)]
    X=build_features(df,pace); base=pre_weekend_cols(X); X=add_history(X,source)
    augmented=base+NEW; W.assert_pre_weekend(augmented)
    W.MODELS.update(
        standings=lambda tr,te,F:(-te.d_season_pts.fillna(0)*100+te.d10_order.fillna(11)*1e-3).to_numpy(),
        pre_logit=lambda tr,te,F:W.m_logit_pod(tr,te,base,C=.03,hl=100),
        win_logit=lambda tr,te,F:win_score(tr,te,base),
        consistency_win=lambda tr,te,F:win_score(tr,te,augmented),
        consistency_null=lambda tr,te,F:win_score(tr,te,augmented,shuffled=True,seed=214))
    R,P=W.walk_forward(X,[2020,2021,2022],['standings','pre_logit','win_logit','consistency_win','consistency_null'],F=augmented,ens={'ens':['standings','pre_logit']},tie=None)
    R.to_csv(ROOT/'results/e214_inner_per_race.csv',index=False)
    P.to_csv(ROOT/'results/e214_inner_predictions.csv',index=False)
    X[['season','round','code','k']+NEW].to_csv(ROOT/'results/e214_features.csv',index=False)
    comparison={}; candidate=R[R.model.eq('consistency_win')].set_index(['season','round']).winner
    for name in ('standings','ens','win_logit'):
        ref=R[R.model.eq(name)].set_index(['season','round']).winner
        delta,ci,p=W.paired_boot(candidate,ref)
        comparison[name]=dict(difference=float(delta),ci=ci.tolist(),p_nonpositive=float(p))
    hits=R.groupby('model').winner.sum().astype(int).to_dict()
    summary=dict(hits=hits,overlap=R.groupby('model').podium.mean().to_dict(),paired=comparison,
        gate=bool(hits['consistency_win']>=37 and all(v['ci'][0]>0 for v in comparison.values()) and hits['consistency_win']>hits['consistency_null']))
    (ROOT/'results/e214_inner_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('RESULT',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
