"""E208: same-compound Q1 ideal-lap history versus matched actual lap history."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.e168_winner import win_score
from f1.data import DATA,ROOT,load_results
from f1.features import build_features,pre_weekend_cols

NEW=['prior_team_ideal_q1','prior_team_actual_q1']
DRY=['HYPERSOFT','ULTRASOFT','SUPERSOFT','SOFT','MEDIUM','HARD','SUPERHARD']


def measure(q1):
    assert not q1.duplicated(['Driver','LapNumber']).any()
    data=q1.copy()
    for name in ['LapTime','Sector1Time','Sector2Time','Sector3Time']:
        data[name]=data[name].dt.total_seconds()
    sectors=['Sector1Time','Sector2Time','Sector3Time']
    good=data.PitInTime.isna()&data.PitOutTime.isna()&data.TrackStatus.astype(str).eq('1')&data.IsAccurate.fillna(False)&data.Compound.isin(DRY)
    good&=np.isfinite(data[['LapTime']+sectors]).all(axis=1)&data[['LapTime']+sectors].gt(0).all(axis=1)
    good&=np.isclose(data[sectors].sum(axis=1),data.LapTime,atol=.010,rtol=0)
    if 'Deleted' in data:
        good&=~data.Deleted.fillna(False).astype(bool)
    data=data[good]
    rows=[]
    for driver,g in data.groupby('Driver',sort=True):
        ideals=[float(part[sectors].min().sum()) for _,part in g.groupby('Compound')]
        ideal=min(ideals); best=float(g.LapTime.min())
        assert ideal<=best+.010
        rows.append(dict(code=driver,ideal_seconds=ideal,best_seconds=best,eligible_laps=len(g)))
    out=pd.DataFrame(rows,columns=['code','ideal_seconds','best_seconds','eligible_laps'])
    for field,column in [('ideal_seconds','ideal_relative'),('best_seconds','actual_relative')]:
        out[column]=(out[field]-out[field].median())/out[field].min()*100 if len(out)>=10 else np.nan
    return out


def add_history(X,source):
    assert not source.duplicated(['season','round','code']).any()
    history=source.merge(X[['season','round','code','team','k']],on=['season','round','code'],validate='one_to_one')
    assert len(history)==len(source)
    teams=history.groupby(['k','team'],sort=True)[['ideal_relative','actual_relative']].mean().reset_index()
    out=X.copy()
    for feature,field in zip(NEW,['ideal_relative','actual_relative']):
        out[feature]=0.
        for k,current in X.groupby('k',sort=True):
            past=teams[teams.k.lt(k)&teams[field].notna()]
            w=.5**((k-past.k)/10)
            rate=(w*past[field]).groupby(past.team).sum()/(5+w.groupby(past.team).sum())
            out.loc[current.index,feature]=current.team.map(rate).fillna(0.)
    return out


def extract():
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True)
    entries=load_results().loc[lambda d:d.season.between(2018,2022),['season','round']].drop_duplicates()
    parts=[]; reports=[]
    for season,round_ in entries.itertuples(index=False,name=None):
        report=dict(season=int(season),round=int(round_),available=False)
        try:
            with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('Q1 extraction HTTP forbidden')):
                session=fastf1.get_session(int(season),int(round_),'Q')
                session.load(laps=True,telemetry=False,weather=False,messages=False)
                q1,_,_=session.laps.split_qualifying_sessions()
            if q1 is None:
                raise ValueError('No Q1 laps')
        except Exception as exc:
            report['error_type']=type(exc).__name__
        else:
            measured=measure(pd.DataFrame(q1)); parts.append(measured.assign(season=season,round=round_))
            report.update(available=True,drivers=len(measured),supported=len(measured)>=10,
                eligible_laps=int(measured.eligible_laps.sum()))
        reports.append(report); print('source',json.dumps(report),flush=True)
    source=pd.concat(parts,ignore_index=True)
    available=sum(r['available'] for r in reports); supported=sum(r.get('supported',False) for r in reports)
    coverage=float(source.ideal_relative.notna().sum()/(20*len(entries)))
    gate=bool(available>=95 and supported>=80 and coverage>=.70)
    source.to_csv(ROOT/'results/e208_q1_source.csv',index=False)
    report=dict(sessions=reports,available=available,supported=supported,entrant_coverage=coverage,coverage_gate=gate)
    (ROOT/'results/e208_source_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    assert gate,'Registered Q1 source coverage failed; no model fit'
    return source


def main():
    source=extract()
    df=load_results().loc[lambda d:d.season.le(2022)]
    pace=pd.read_csv(DATA/'racepace_ff1.csv').loc[lambda d:d.season.le(2022)]
    X=build_features(df,pace); base=pre_weekend_cols(X); X=add_history(X,source)
    augmented=base+[NEW[0]]; W.assert_pre_weekend(base+NEW)
    W.MODELS.update(
        standings=lambda tr,te,F:(-te.d_season_pts.fillna(0)*100+te.d10_order.fillna(11)*1e-3).to_numpy(),
        pre_logit=lambda tr,te,F:W.m_logit_pod(tr,te,base,C=.03,hl=100),
        win_logit=lambda tr,te,F:win_score(tr,te,base),
        actual_win=lambda tr,te,F:win_score(tr,te,base+[NEW[1]]),
        ideal_win=lambda tr,te,F:win_score(tr,te,augmented),
        ideal_null=lambda tr,te,F:win_score(tr,te,augmented,shuffled=True,seed=208))
    R,P=W.walk_forward(X,[2020,2021,2022],['standings','pre_logit','win_logit','actual_win','ideal_win','ideal_null'],F=augmented,ens={'ens':['standings','pre_logit']},tie=None)
    R.to_csv(ROOT/'results/e208_inner_per_race.csv',index=False)
    P.to_csv(ROOT/'results/e208_inner_predictions.csv',index=False)
    X[['season','round','code','k']+NEW].to_csv(ROOT/'results/e208_features.csv',index=False)
    comparison={}; candidate=R[R.model.eq('ideal_win')].set_index(['season','round']).winner
    for name in ('standings','ens','win_logit','actual_win'):
        ref=R[R.model.eq(name)].set_index(['season','round']).winner
        delta,ci,p=W.paired_boot(candidate,ref)
        comparison[name]=dict(difference=float(delta),ci=ci.tolist(),p_nonpositive=float(p))
    hits=R.groupby('model').winner.sum().astype(int).to_dict()
    summary=dict(hits=hits,overlap=R.groupby('model').podium.mean().to_dict(),paired=comparison,
        gate=bool(hits['ideal_win']>=37 and all(v['ci'][0]>0 for v in comparison.values()) and hits['ideal_win']>hits['ideal_null']))
    (ROOT/'results/e208_inner_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('RESULT',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
