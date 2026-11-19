"""E235 one earlier-driver launch feature in the frozen winner classifier."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.e168_winner import win_score
from experiments.e190_traffic_feasibility import cached_session
from experiments.e233_launch_source import event
from f1.data import DATA,ROOT,load_results
from f1.features import build_features,pre_weekend_cols

NEW=['prior_driver_launch']
SOURCE=ROOT/'results/e235_source_status.json'


def observations(reports,entries):
    assert entries.season.between(2018,2022).all()
    assert not entries.duplicated(['season','round','code']).any()
    assert len({(r['season'],r['round']) for r in reports})==len(reports)
    lookup={}; legal=set(map(tuple,entries[['season','round','code']].to_numpy()))
    for r in reports:
        assert 2018<=r['season']<=2022
        assert (r['season'],r['round']) in {(x[0],x[1]) for x in legal}
        for c in r.get('rows',[]):
            key=(r['season'],r['round'],c['driver'])
            assert key in legal and key not in lookup
            value=c['relative20'] if r['gate'] and c['supported'] else np.nan
            assert pd.isna(value) or np.isfinite(value)
            lookup[key]=value
    table=entries[['season','round','code']].copy()
    table['response']=[lookup.get(tuple(k),np.nan) for k in table.to_numpy()]
    return table.reset_index(drop=True)


def add_history(X,table):
    assert table.season.between(2018,2022).all()
    assert not table.duplicated(['season','round','code']).any()
    events=X[['season','round','k']].drop_duplicates()
    history=table.merge(events,on=['season','round'],validate='many_to_one')
    assert len(history)==len(table)
    out=X.copy(); out[NEW[0]]=0.
    for k,current in X.groupby('k',sort=True):
        past=history[history.k.lt(k)&history.response.notna()]
        weights=.5**((k-past.k)/10)
        values=(weights*past.response).groupby(past.code).sum()/(5+weights.groupby(past.code).sum())
        out.loc[current.index,NEW[0]]=current.code.map(values).fillna(0.)
    return out


def extract():
    import fastf1
    import f1.ff1
    inventory=json.loads((ROOT/'results/e212_source_status.json').read_text())
    assert len(inventory)==103 and sum(r['available'] for r in inventory)==99
    reports=json.loads(SOURCE.read_text()) if SOURCE.exists() else []
    done={(r['season'],r['round']) for r in reports}
    assert len(done)==len(reports)
    assert done.issubset({(r['season'],r['round']) for r in inventory})
    fastf1.Cache.offline_mode(True)
    for receipt in inventory:
        season,round_=receipt['season'],receipt['round']
        if (season,round_) in done:
            continue
        report=dict(season=season,round=round_,available=False,gate=False)
        if not receipt['available']:
            report['reason']='unavailable_in_frozen_cache_inventory'
        else:
            try:
                with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('launch winner HTTP forbidden')):
                    session=cached_session(fastf1,season,round_)
                    cars=fastf1._api.car_data(session.api_path)
            except Exception as exc:
                report['error_type']=type(exc).__name__
            else:
                report.update(available=True,**event(pd.DataFrame(session.laps),cars))
        reports.append(report)
        SOURCE.write_text(json.dumps(reports,indent=2)+'\n')
        print('SOURCE',season,round_,report['available'],report.get('supported',0),report['gate'],flush=True)
    entries=load_results().loc[lambda d:d.season.between(2018,2022),['season','round','code']]
    table=observations(reports,entries)
    available=sum(r['available'] for r in reports); supported=sum(r['gate'] for r in reports)
    coverage=float(table.response.notna().mean())
    summary=dict(sessions=len(reports),available=available,supported_events=supported,driver_coverage=coverage,
        gate=bool(available>=95 and supported>=80 and coverage>=.8))
    table.to_csv(ROOT/'results/e235_launch_observations.csv',index=False)
    (ROOT/'results/e235_source_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    assert summary['gate'],'Registered launch source gate failed; no winner fit'
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
        launch_win=lambda tr,te,F:win_score(tr,te,augmented),
        launch_null=lambda tr,te,F:win_score(tr,te,augmented,shuffled=True,seed=235))
    R,P=W.walk_forward(X,[2020,2021,2022],['standings','pre_logit','win_logit','launch_win','launch_null'],F=augmented,ens={'ens':['standings','pre_logit']},tie=None)
    R.to_csv(ROOT/'results/e235_inner_per_race.csv',index=False)
    P.to_csv(ROOT/'results/e235_inner_predictions.csv',index=False)
    X[['season','round','code','k']+NEW].to_csv(ROOT/'results/e235_features.csv',index=False)
    comparison={}; candidate=R[R.model.eq('launch_win')].set_index(['season','round']).winner
    for name in ('standings','ens','win_logit'):
        ref=R[R.model.eq(name)].set_index(['season','round']).winner
        delta,ci,p=W.paired_boot(candidate,ref)
        comparison[name]=dict(difference=float(delta),ci=ci.tolist(),p_nonpositive=float(p))
    hits=R.groupby('model').winner.sum().astype(int).to_dict()
    summary=dict(hits=hits,overlap=R.groupby('model').podium.mean().to_dict(),paired=comparison,
        gate=bool(hits['launch_win']>=37 and all(v['ci'][0]>0 for v in comparison.values()) and hits['launch_win']>hits['launch_null']))
    (ROOT/'results/e235_inner_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('RESULT',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
