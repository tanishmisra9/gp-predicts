"""E207: fixed earlier-driver restart gains for exact winner prediction."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.e168_winner import win_score
from experiments.e190_traffic_feasibility import cached_session
from experiments.e206_restart_source import COLS,extract_restarts
from f1.data import DATA,ROOT,load_results
from f1.features import build_features,pre_weekend_cols

NEW=['prior_restart_gain']


def add_history(X,source):
    assert not source.duplicated(['season','round','restart','driver']).any()
    assert np.isfinite(source.gain).all()
    events=source.groupby(['season','round','driver'],sort=True).gain.mean().reset_index().rename(columns={'driver':'code'})
    history=events.merge(X[['season','round','code','k']],on=['season','round','code'],validate='one_to_one')
    assert len(history)==len(events)
    out=X.copy(); out[NEW[0]]=0.
    for k,current in X.groupby('k',sort=True):
        past=history[history.k.lt(k)]
        w=.5**((k-past.k)/10)
        numerator=(w*past.gain).groupby(past.code).sum()
        denominator=w.groupby(past.code).sum()
        rate=numerator/(5+denominator)
        out.loc[current.index,NEW[0]]=current.code.map(rate).fillna(0.)
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
            session=cached_session(fastf1,int(season),int(round_))
            laps=pd.DataFrame(session.laps); flags=pd.DataFrame(session.track_status)
            with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('restart extraction HTTP forbidden')):
                _,stream,_=fastf1._api._extended_timing_data(session.api_path)
        except Exception as exc:
            report['error_type']=type(exc).__name__
        else:
            identities=laps[['DriverNumber','Driver']].drop_duplicates()
            assert not identities.DriverNumber.duplicated().any()
            mapping=identities.assign(DriverNumber=identities.DriverNumber.astype(str)).set_index('DriverNumber').Driver
            stream=stream.copy(); stream.Driver=stream.Driver.astype(str).map(mapping)
            assert stream.Driver.notna().all()
            measured,events=extract_restarts(laps,flags,stream)
            parts.append(measured.assign(season=season,round=round_))
            report.update(available=True,events=events,rows=len(measured),usable_restarts=sum(e['usable'] for e in events))
        reports.append(report); print('source',json.dumps(report),flush=True)
    source=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=COLS+['season','round'])
    available=sum(r['available'] for r in reports)
    restarts=sum(r.get('usable_restarts',0) for r in reports)
    races=sum(r.get('usable_restarts',0)>0 for r in reports)
    gate=bool(available>=95 and restarts>=20 and races>=15 and len(source)>=300)
    source.to_csv(ROOT/'results/e207_restart_source.csv',index=False)
    summary=dict(sessions=reports,available=available,usable_restarts=restarts,races=races,opportunities=len(source),coverage_gate=gate)
    (ROOT/'results/e207_source_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    assert gate,'Registered restart source coverage failed; no winner fit'
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
        restart_win=lambda tr,te,F:win_score(tr,te,augmented),
        restart_null=lambda tr,te,F:win_score(tr,te,augmented,shuffled=True,seed=207))
    R,P=W.walk_forward(X,[2020,2021,2022],['standings','pre_logit','win_logit','restart_win','restart_null'],F=augmented,ens={'ens':['standings','pre_logit']},tie=None)
    R.to_csv(ROOT/'results/e207_inner_per_race.csv',index=False)
    P.to_csv(ROOT/'results/e207_inner_predictions.csv',index=False)
    X[['season','round','code','k']+NEW].to_csv(ROOT/'results/e207_features.csv',index=False)
    comparison={}; candidate=R[R.model.eq('restart_win')].set_index(['season','round']).winner
    for name in ('standings','ens','win_logit'):
        ref=R[R.model.eq(name)].set_index(['season','round']).winner
        delta,ci,p=W.paired_boot(candidate,ref)
        comparison[name]=dict(difference=float(delta),ci=ci.tolist(),p_nonpositive=float(p))
    hits=R.groupby('model').winner.sum().astype(int).to_dict()
    summary=dict(hits=hits,overlap=R.groupby('model').podium.mean().to_dict(),paired=comparison,
        gate=bool(hits['restart_win']>=37 and all(v['ci'][0]>0 for v in comparison.values()) and hits['restart_win']>hits['restart_null']))
    (ROOT/'results/e207_inner_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('RESULT',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
