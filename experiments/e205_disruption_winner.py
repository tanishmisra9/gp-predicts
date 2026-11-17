"""E205: one strictly prior team lap-disruption rate for exact winners."""
import json

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.e168_winner import win_score
from experiments.e190_traffic_feasibility import cached_session
from f1.data import DATA,ROOT,load_results
from f1.features import build_features,pre_weekend_cols

NEW=['prior_team_disruption']


def disruptions(laps):
    assert not laps.duplicated(['Driver','LapNumber']).any()
    valid=laps.LapNumber.gt(1)&laps.LapTime.notna()&laps.PitInTime.isna()&laps.PitOutTime.isna()
    valid&=laps.TrackStatus.astype(str).eq('1')&laps.IsAccurate.fillna(False)
    clean=laps[valid].copy(); clean['seconds']=clean.LapTime.dt.total_seconds()
    clean=clean[np.isfinite(clean.seconds)&clean.seconds.gt(0)]
    rows=[]
    for driver in sorted(laps.Driver.dropna().unique()):
        g=clean[clean.Driver.eq(driver)].sort_values('LapNumber').set_index('LapNumber')
        eligible=0; count=0
        for n,r in g.iterrows():
            if n-1 not in g.index or n+1 not in g.index:
                continue
            window=g.loc[[n-1,n,n+1]]
            if pd.isna(r.Stint) or not window.Stint.eq(r.Stint).all() or r.Compound not in ('SOFT','MEDIUM','HARD') or not window.Compound.eq(r.Compound).all():
                continue
            if not np.allclose(np.diff(window.TyreLife.to_numpy(float)),1.):
                continue
            eligible+=1
            baseline=(g.loc[n-1,'seconds']+g.loc[n+1,'seconds'])/2
            count+=int(r.seconds>=1.02*baseline)
        rows.append(dict(code=driver,eligible=eligible,disruptions=count))
    return pd.DataFrame(rows,columns=['code','eligible','disruptions'])


def add_history(X,source):
    assert not source.duplicated(['season','round','code']).any()
    assert source.eligible.ge(source.disruptions).all() and source.disruptions.ge(0).all()
    metadata=X[['season','round','k','code','team']]
    history=source.merge(metadata,on=['season','round','code'],validate='one_to_one')
    assert len(history)==len(source)
    out=X.copy(); out[NEW[0]]=np.nan
    for k,current in X.groupby('k',sort=True):
        past=history[history.k.lt(k)]
        w=.5**((k-past.k)/10)
        global_rate=(float((w*past.disruptions).sum())+.5)/(float((w*past.eligible).sum())+1)
        numer=(w*past.disruptions).groupby(past.team).sum()
        denom=(w*past.eligible).groupby(past.team).sum()
        rate=(numer+100*global_rate)/(denom+100)
        out.loc[current.index,NEW[0]]=current.team.map(rate).fillna(global_rate)
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
        except Exception as exc:
            report['error_type']=type(exc).__name__
        else:
            part=disruptions(pd.DataFrame(session.laps)).assign(season=season,round=round_)
            parts.append(part); report.update(available=True,drivers=len(part),eligible=int(part.eligible.sum()),disruptions=int(part.disruptions.sum()))
        reports.append(report); print('source',json.dumps(report),flush=True)
    source=pd.concat(parts,ignore_index=True)
    gate=bool(sum(r['available'] for r in reports)>=95 and source.eligible.ge(3).mean()>=.8)
    source.to_csv(ROOT/'results/e205_disruption_source.csv',index=False)
    (ROOT/'results/e205_source_summary.json').write_text(json.dumps(dict(events=reports,coverage_gate=gate),indent=2)+'\n')
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
        disruption_win=lambda tr,te,F:win_score(tr,te,augmented),
        disruption_null=lambda tr,te,F:win_score(tr,te,augmented,shuffled=True,seed=205))
    R,P=W.walk_forward(X,[2020,2021,2022],['standings','pre_logit','win_logit','disruption_win','disruption_null'],F=augmented,ens={'ens':['standings','pre_logit']},tie=None)
    R.to_csv(ROOT/'results/e205_inner_per_race.csv',index=False)
    P.to_csv(ROOT/'results/e205_inner_predictions.csv',index=False)
    X[['season','round','code','k']+NEW].to_csv(ROOT/'results/e205_features.csv',index=False)
    comparison={}; candidate=R[R.model.eq('disruption_win')].set_index(['season','round']).winner
    for name in ('standings','ens','win_logit'):
        ref=R[R.model.eq(name)].set_index(['season','round']).winner
        delta,ci,p=W.paired_boot(candidate,ref)
        comparison[name]=dict(difference=float(delta),ci=ci.tolist(),p_nonpositive=float(p))
    hits=R.groupby('model').winner.sum().astype(int).to_dict()
    summary=dict(hits=hits,overlap=R.groupby('model').podium.mean().to_dict(),paired=comparison,
        gate=bool(hits['disruption_win']>=37 and all(v['ci'][0]>0 for v in comparison.values()) and hits['disruption_win']>hits['disruption_null']))
    (ROOT/'results/e205_inner_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('RESULT',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
