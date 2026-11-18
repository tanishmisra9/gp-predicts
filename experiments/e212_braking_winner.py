"""E212: fixed earlier team braking signature for pre-weekend winners."""
import json
import os
from unittest.mock import patch

import fastf1
import numpy as np
import pandas as pd
import requests

import f1.walkforward as W
from experiments.e168_winner import win_score
from experiments.e190_traffic_feasibility import cached_session
from experiments.e196_weather_source import restricted_send
from experiments.e211_braking_signature import source_segments
from f1.data import DATA,ROOT,load_results
from f1.features import build_features,pre_weekend_cols

NEW=['prior_team_braking']
SOURCE=ROOT/'results/e212_braking_laps.csv'
STATUS=ROOT/'results/e212_source_status.json'


def allowed_urls(path,season):
    assert 2018<=season<=2022 and path.startswith(f'/static/{season}/') and path.endswith('_Race/')
    return {host+path+'CarData.z.jsonStream' for host in ('https://livetiming.formula1.com','https://livetiming-mirror.fastf1.dev')}


def observations(source,entries):
    assert source.season.between(2018,2022).all() and entries.season.between(2018,2022).all()
    assert not source.duplicated(['season','round','driver','lap']).any()
    assert not entries.duplicated(['season','round','code']).any()
    assert np.isfinite(source.duration.to_numpy(float)).all()
    legal=entries[['season','round','code']].rename(columns={'code':'driver'})
    assert len(source.merge(legal,on=['season','round','driver'],validate='many_to_one'))==len(source)
    means=source.groupby(['season','round','driver'],sort=True).duration.agg(['mean','size'])
    rows=[]
    for (season,round_,team),event in entries.groupby(['season','round','team'],sort=True):
        values=[]
        for code in sorted(event.code.unique()):
            key=(season,round_,code)
            values.append(float(means.loc[key,'mean']) if key in means.index and means.loc[key,'size']>=10 else np.nan)
        response=float(np.mean(values)) if len(values)==2 and np.isfinite(values).all() else np.nan
        rows.append(dict(season=int(season),round=int(round_),team=team,response=response))
    table=pd.DataFrame(rows)
    group=table.groupby(['season','round']).response
    table['centered_response']=(table.response-group.transform('mean')).where(group.transform('count').ge(6))
    return table


def add_history(X,table):
    assert not table.duplicated(['season','round','team']).any()
    events=X[['season','round','k']].drop_duplicates()
    history=table.merge(events,on=['season','round'],validate='many_to_one')
    assert len(history)==len(table)
    out=X.copy(); out[NEW[0]]=0.
    for k,current in X.groupby('k',sort=True):
        past=history[history.k.lt(k)&history.centered_response.notna()]
        w=.5**((k-past.k)/10)
        rate=(w*past.centered_response).groupby(past.team).sum()/(5+w.groupby(past.team).sum())
        out.loc[current.index,NEW[0]]=current.team.map(rate).fillna(0.)
    return out


def extract():
    import f1.ff1
    from fastf1.internals.f1account import get_token
    events=load_results().loc[lambda d:d.season.between(2018,2022),['season','round']].drop_duplicates()
    reports=json.loads(STATUS.read_text()) if STATUS.exists() else []
    done={(r['season'],r['round']) for r in reports}
    assert len(done)==len(reports) and done.issubset(set(map(tuple,events.to_numpy())))
    if reports:
        assert SOURCE.exists()
    source=pd.read_csv(SOURCE,float_precision='round_trip') if SOURCE.exists() else pd.DataFrame(columns=['driver','lap','duration','season','round'])
    if len(source):
        source=source[[tuple(x) in done for x in source[['season','round']].to_numpy()]].copy()
    if len(done)<len(events):
        if not all(os.environ.get(k) for k in ('F1_USERNAME','F1_PASSWORD')):
            raise RuntimeError('Missing existing F1 credentials')
        fastf1.Cache.offline_mode(False)
        try:
            get_token(method='password')
        except Exception as exc:
            (ROOT/'results/e212_acquisition_error.json').write_text(json.dumps(dict(error_type=type(exc).__name__))+'\n')
            raise RuntimeError('Existing account authentication failed') from None
    for season,round_ in events.itertuples(index=False,name=None):
        if (season,round_) in done:
            continue
        report=dict(season=int(season),round=int(round_),available=False,requests=[])
        fastf1.Cache.offline_mode(True)
        try:
            session=cached_session(fastf1,int(season),int(round_)); laps=pd.DataFrame(session.laps)
        except Exception as exc:
            report['error_type']=type(exc).__name__
        else:
            fastf1.Cache.offline_mode(False)
            try:
                with patch('requests.adapters.HTTPAdapter.send',new=restricted_send(requests.adapters.HTTPAdapter.send,allowed_urls(session.api_path,int(season)),report['requests'])):
                    cars=fastf1._api.car_data(session.api_path)
            except Exception as exc:
                report['error_type']=type(exc).__name__
            else:
                segments=source_segments(laps,cars)
                lap_values=segments.groupby(['driver','lap'],sort=True).duration.median().reset_index().assign(season=season,round=round_)
                source=pd.concat([source,lap_values],ignore_index=True)
                report.update(available=True,segments=len(segments),driver_laps=len(lap_values),car_rows=sum(len(c) for c in cars.values()),padded_rows=sum(int(c.Time.isna().sum()) for c in cars.values()))
        reports.append(report); source.to_csv(SOURCE,index=False)
        STATUS.write_text(json.dumps(reports,indent=2)+'\n')
        print('SOURCE',json.dumps(report),flush=True)
    entries=load_results().loc[lambda d:d.season.between(2018,2022),['season','round','code','team']]
    table=observations(source,entries)
    support=int(table.groupby(['season','round']).centered_response.count().ge(6).sum())
    coverage=float(table.centered_response.notna().mean())
    available=sum(r['available'] for r in reports)
    gate=bool(available>=95 and support>=80 and coverage>=.8)
    table.to_csv(ROOT/'results/e212_braking_observations.csv',index=False)
    summary=dict(sessions=len(reports),available=available,supported_events=support,team_coverage=coverage,gate=gate)
    (ROOT/'results/e212_source_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    assert gate,'Registered braking source coverage gate failed; no winner fit'
    return table


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
        braking_win=lambda tr,te,F:win_score(tr,te,augmented),
        braking_null=lambda tr,te,F:win_score(tr,te,augmented,shuffled=True,seed=212))
    R,P=W.walk_forward(X,[2020,2021,2022],['standings','pre_logit','win_logit','braking_win','braking_null'],F=augmented,ens={'ens':['standings','pre_logit']},tie=None)
    R.to_csv(ROOT/'results/e212_inner_per_race.csv',index=False)
    P.to_csv(ROOT/'results/e212_inner_predictions.csv',index=False)
    X[['season','round','code','k']+NEW].to_csv(ROOT/'results/e212_features.csv',index=False)
    comparison={}; candidate=R[R.model.eq('braking_win')].set_index(['season','round']).winner
    for name in ('standings','ens','win_logit'):
        ref=R[R.model.eq(name)].set_index(['season','round']).winner
        delta,ci,p=W.paired_boot(candidate,ref)
        comparison[name]=dict(difference=float(delta),ci=ci.tolist(),p_nonpositive=float(p))
    hits=R.groupby('model').winner.sum().astype(int).to_dict()
    summary=dict(hits=hits,overlap=R.groupby('model').podium.mean().to_dict(),paired=comparison,
        gate=bool(hits['braking_win']>=37 and all(v['ci'][0]>0 for v in comparison.values()) and hits['braking_win']>hits['braking_null']))
    (ROOT/'results/e212_inner_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('RESULT',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
