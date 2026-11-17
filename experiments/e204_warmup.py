"""E204: post-stop first-full-lap response; fixed 2019 measurement only."""
import json

import numpy as np
import pandas as pd

from experiments.e190_traffic_feasibility import cached_session,start_exposure
from experiments.e191_traffic_repeatability import past_response,cluster_interval
from f1.data import ROOT,load_results


def blocks(laps):
    assert not laps.duplicated(['Driver','LapNumber']).any()
    clean=start_exposure(laps).set_index(['Driver','LapNumber'],verify_integrity=True)
    rows=[]
    for out in laps[laps.PitOutTime.notna()].itertuples():
        prior=laps[laps.Driver.eq(out.Driver)&laps.LapNumber.eq(out.LapNumber-1)]
        if len(prior)!=1 or pd.isna(prior.PitInTime.iloc[0]):
            continue
        pit_seconds=(out.PitOutTime-prior.PitInTime.iloc[0]).total_seconds()
        if not 10<pit_seconds<60:
            continue
        numbers=np.arange(int(out.LapNumber)+1,int(out.LapNumber)+5)
        keys=[(out.Driver,int(n)) for n in numbers]
        if not all(k in clean.index for k in keys):
            continue
        g=clean.loc[keys]
        valid=g.Stint.eq(out.Stint).all() and g.Compound.eq(out.Compound).all()
        valid=valid and out.Compound in ('SOFT','MEDIUM','HARD') and g.Position.notna().all() and g.Position.nunique()==1
        valid=valid and np.allclose(np.diff(g.TyreLife.to_numpy(float)),1.)
        if not valid:
            continue
        times=g.seconds.to_numpy(float)
        contrast=float(times@np.array([1.,-4/3,-1/3,2/3])/times.mean()*100)
        rows.append(dict(driver=out.Driver,pitout_lap=int(out.LapNumber),first_full_lap=int(numbers[0]),
            stint=out.Stint,compound=out.Compound,seconds0=times[0],seconds1=times[1],
            seconds2=times[2],seconds3=times[3],contrast=contrast))
    return pd.DataFrame(rows,columns=['driver','pitout_lap','first_full_lap','stint','compound',
        'seconds0','seconds1','seconds2','seconds3','contrast'])


def measurements(source,entries):
    rows=[]
    for (season,round_,team),event in entries.groupby(['season','round','team'],sort=True):
        p=source[source.season.eq(season)&source['round'].eq(round_)&source.team.eq(team)]
        drivers=sorted(event.code.unique()); values=p.groupby('driver').contrast.median()
        a=values.get(drivers[0],np.nan) if len(drivers)==2 else np.nan
        b=values.get(drivers[1],np.nan) if len(drivers)==2 else np.nan
        full=(a+b)/2 if np.isfinite([a,b]).all() else np.nan
        rows.append(dict(season=int(season),round=int(round_),team=team,blocks=len(p),
            half0=a,half1=b,response=full))
    table=pd.DataFrame(rows)
    for col in ('response','half0','half1'):
        group=table.groupby(['season','round'])[col]
        table['centered_'+col]=(table[col]-group.transform('mean')).where(group.transform('count').ge(6))
    return table


def evaluate(table):
    forecast=past_response(table)
    halves=table.dropna(subset=['centered_half0','centered_half1'])
    eligible=forecast[forecast.past_count.ge(3)&forecast.centered_response.notna()].copy()
    races=int(table.groupby('round').centered_response.count().ge(6).sum())
    result=dict(valid_races=races,split_rows=len(halves),temporal_rows=len(eligible),
        coverage_gate=bool(races>=12 and len(halves)>=60 and len(eligible)>=60),gate=False)
    if not result['coverage_gate']:
        return result,forecast
    split,split_ci=cluster_interval(halves,'centered_half0','centered_half1')
    temporal,temporal_ci=cluster_interval(eligible,'prediction','centered_response')
    eligible['mse_gain']=eligible.centered_response**2-(eligible.prediction-eligible.centered_response)**2
    gain,gain_ci=cluster_interval(eligible,'mse_gain')
    null=table.copy(); rng=np.random.default_rng(204)
    for _,ids in null[null.centered_response.notna()].groupby('round').groups.items():
        null.loc[ids,'centered_response']=null.loc[rng.permutation(ids),'centered_response'].to_numpy()
    null=past_response(null)
    eligible['null_gain']=(null.prediction.reindex(eligible.index)-eligible.centered_response)**2-(eligible.prediction-eligible.centered_response)**2
    null_gain,null_ci=cluster_interval(eligible,'null_gain')
    result.update(split_correlation=split,split_ci=split_ci.tolist(),temporal_correlation=temporal,
        temporal_ci=temporal_ci.tolist(),mse_gain=gain,mse_gain_ci=gain_ci.tolist(),
        baseline_mse=float((eligible.centered_response**2).mean()),
        forecast_mse=float(((eligible.prediction-eligible.centered_response)**2).mean()),
        null_mse_gain=null_gain,null_mse_gain_ci=null_ci.tolist())
    result['gate']=bool(split>=.3 and split_ci[0]>0 and temporal>=.15 and gain>0 and gain_ci[0]>0 and null_gain>0 and null_ci[0]>0)
    return result,forecast


def main():
    import fastf1
    import f1.ff1
    entries=load_results().loc[lambda d:d.season.eq(2019),['season','round','code','team']]
    assert set(entries['round'])==set(range(1,22))
    fastf1.Cache.offline_mode(True); parts=[]; status=[]
    for round_ in range(1,22):
        session=cached_session(fastf1,2019,round_)
        part=blocks(pd.DataFrame(session.laps)).assign(season=2019,round=round_)
        part=part.merge(entries[entries['round'].eq(round_)][['code','team']].rename(columns={'code':'driver'}),on='driver',how='left',validate='many_to_one')
        assert part.team.notna().all()
        parts.append(part); status.append(dict(round=round_,blocks=len(part)))
        print('source',round_,len(part),flush=True)
    source=pd.concat(parts,ignore_index=True); table=measurements(source,entries)
    source.to_csv(ROOT/'results/e204_warmup_blocks.csv',index=False)
    table.to_csv(ROOT/'results/e204_warmup_measurements.csv',index=False)
    result,forecast=evaluate(table); result.update(source=status,blocks=len(source))
    forecast.to_csv(ROOT/'results/e204_warmup_forecasts.csv',index=False)
    (ROOT/'results/e204_warmup_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
