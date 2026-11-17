"""E191: fixed 2019 observational traffic-response measurement, cache only."""
import json

import numpy as np
import pandas as pd

from experiments.e190_traffic_feasibility import REQUIRED,cached_session,disjoint_pairs,start_exposure
from f1.data import ROOT,load_results

PAIR_FILE=ROOT/'results/e191_source_pairs.csv'
STATUS_FILE=ROOT/'results/e191_source_status.json'
PAIR_COLUMNS=['driver','lap1','lap2','entering','stint','seconds1','seconds2','completion2','contrast','season','round','team']


def pair_measurements(clean):
    pairs=disjoint_pairs(clean)
    keys=['Driver','LapNumber']
    values=clean[keys+['seconds','Time']]
    for side in (1,2):
        table=values.rename(columns={'Driver':'driver','LapNumber':f'lap{side}','seconds':f'seconds{side}','Time':f'time{side}'})
        pairs=pairs.merge(table,on=['driver',f'lap{side}'],how='left',validate='many_to_one')
    direction=np.where(pairs.entering,1.,-1.)
    pairs['contrast']=(pairs.seconds2-pairs.seconds1)/((pairs.seconds1+pairs.seconds2)/2)*100*direction
    pairs['completion2']=pairs.time2.dt.total_seconds()
    return pairs.drop(columns=['time1','time2'])


def balanced_response(pairs,min_pairs=8,min_each=2):
    entering=pairs.entering.astype(bool)
    if len(pairs)<min_pairs or entering.sum()<min_each or (~entering).sum()<min_each:
        return np.nan
    return .5*(pairs[entering].contrast.mean()+pairs[~entering].contrast.mean())


def team_measurements(pairs,entries):
    rows=[]
    for (season,round_,team),_ in entries.groupby(['season','round','team'],sort=True):
        p=pairs[pairs.season.eq(season)&pairs['round'].eq(round_)&pairs.team.eq(team)].sort_values(['completion2','driver','lap1'])
        half=np.arange(len(p))%2
        rows.append(dict(season=season,round=round_,team=team,pairs=len(p),response=balanced_response(p),
            half0=balanced_response(p.iloc[np.flatnonzero(half==0)],4,1),
            half1=balanced_response(p.iloc[np.flatnonzero(half==1)],4,1)))
    out=pd.DataFrame(rows)
    for col in ('response','half0','half1'):
        group=out.groupby(['season','round'])[col]
        out['centered_'+col]=(out[col]-group.transform('mean')).where(group.transform('count').ge(6))
    return out


def past_response(table,value='centered_response'):
    out=table.copy(); out['prediction']=0.; out['past_count']=0
    rounds=sorted(out['round'].unique()); k={r:i for i,r in enumerate(rounds)}
    for i,row in out.iterrows():
        past=out[out['round'].lt(row['round'])&out.team.eq(row.team)&out[value].notna()]
        weights=.5**((k[row['round']]-past['round'].map(k))/10)
        out.loc[i,'prediction']=(past[value]*weights).sum()/(5+weights.sum())
        out.loc[i,'past_count']=len(past)
    return out


def cluster_interval(frame,x,y=None):
    def statistic(g):
        if y is None:
            return float(g[x].mean())
        return float(g[x].corr(g[y])) if len(g)>1 and g[x].std()>0 and g[y].std()>0 else np.nan
    if frame.empty:
        return np.nan,np.array([np.nan,np.nan])
    groups=[g for _,g in frame.groupby('round',sort=True)]
    rng=np.random.default_rng(191)
    samples=[statistic(pd.concat([groups[i] for i in rng.integers(len(groups),size=len(groups))])) for _ in range(4000)]
    finite=np.asarray(samples)[np.isfinite(samples)]
    return statistic(frame),np.percentile(finite,[2.5,97.5]) if len(finite) else np.array([np.nan,np.nan])


def extract(entries):
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True)
    reports=json.loads(STATUS_FILE.read_text()) if STATUS_FILE.exists() else []
    pairs=pd.read_csv(PAIR_FILE) if PAIR_FILE.exists() else pd.DataFrame(columns=PAIR_COLUMNS)
    done={(r['season'],r['round']) for r in reports}
    for (season,round_),event in entries.groupby(['season','round'],sort=True):
        if (season,round_) in done:
            continue
        report=dict(season=int(season),round=int(round_))
        try:
            session=cached_session(fastf1,int(season),int(round_))
        except Exception as exc:
            report.update(available=False,error_type=type(exc).__name__)
        else:
            missing=sorted(set(REQUIRED)-set(session.laps.columns))
            report.update(available=not missing,missing_fields=missing)
            if not missing:
                measured=pair_measurements(start_exposure(pd.DataFrame(session.laps)))
                measured=measured.assign(season=int(season),round=int(round_)).merge(
                    event[['code','team']].rename(columns={'code':'driver'}),on='driver',how='left',validate='many_to_one')
                assert measured.team.notna().all()
                pairs=pd.concat([pairs,measured[PAIR_COLUMNS]],ignore_index=True)
                report['pairs']=len(measured)
        reports.append(report)
        pairs.to_csv(PAIR_FILE,index=False)
        STATUS_FILE.write_text(json.dumps(reports,indent=2)+'\n')
        print('source',json.dumps(report),flush=True)
    assert len(reports)==21 and all(r['season']==2019 for r in reports)
    return pairs,reports


def main():
    entries=load_results().loc[lambda d:d.season.eq(2019),['season','round','code','team']]
    assert entries['round'].nunique()==21
    pairs,reports=extract(entries)
    table=team_measurements(pairs,entries)
    table.to_csv(ROOT/'results/e191_team_measurements.csv',index=False)
    valid_races=table.groupby('round').centered_response.count().ge(6).sum()
    halves=table.dropna(subset=['centered_half0','centered_half1'])
    split,split_ci=cluster_interval(halves,'centered_half0','centered_half1')
    forecast=past_response(table)
    eligible=forecast[forecast.past_count.ge(3)&forecast.centered_response.notna()].copy()
    eligible['mse_gain']=eligible.centered_response**2-(eligible.prediction-eligible.centered_response)**2
    gain,gain_ci=cluster_interval(eligible,'mse_gain')
    temporal,temporal_ci=cluster_interval(eligible,'prediction','centered_response')
    null=table.copy(); rng=np.random.default_rng(191)
    for _,ids in null[null.centered_response.notna()].groupby('round').groups.items():
        null.loc[ids,'centered_response']=null.loc[rng.permutation(ids),'centered_response'].to_numpy()
    null=past_response(null)
    null_pred=null.prediction.reindex(eligible.index)
    null_mse=float(((null_pred-eligible.centered_response)**2).mean())
    result=dict(sessions_available=sum(r['available'] for r in reports),total_sessions=21,pairs=len(pairs),
        valid_races=int(valid_races),split_pairs=len(halves),split_correlation=split,split_ci=split_ci.tolist(),
        temporal_rows=len(eligible),temporal_correlation=temporal,temporal_ci=temporal_ci.tolist(),
        mse_gain=gain,mse_gain_ci=gain_ci.tolist(),baseline_mse=float((eligible.centered_response**2).mean()),
        forecast_mse=float(((eligible.prediction-eligible.centered_response)**2).mean()),null_mse=null_mse)
    result['gate']=bool(valid_races>=12 and len(halves)>=60 and split>=.30 and split_ci[0]>0 and len(eligible)>=60 and temporal>=.15 and gain>0 and gain_ci[0]>0)
    print('MEASUREMENT',json.dumps(result),flush=True)
    (ROOT/'results/e191_measurement_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    forecast.to_csv(ROOT/'results/e191_temporal_forecasts.csv',index=False)


if __name__=='__main__':
    main()
