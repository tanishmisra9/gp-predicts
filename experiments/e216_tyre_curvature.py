"""E216: signed nonlinear tyre-age pace curvature, fixed 2019 measurement."""
import json

import numpy as np
import pandas as pd

from experiments.e190_traffic_feasibility import cached_session,start_exposure
from experiments.e191_traffic_repeatability import cluster_interval
from experiments.e204_warmup import evaluate
from f1.data import ROOT,load_results

COLS=['driver','stint','compound','n','n0','n1','response','half0','half1']


def curvature(g,minimum):
    if len(g)<minimum or g.TyreLife.nunique()!=len(g) or g.TyreLife.max()-g.TyreLife.min()<10:
        return np.nan
    age=g.TyreLife.to_numpy(float); y=g.residual.to_numpy(float)
    assert np.isfinite(age).all() and np.isfinite(y).all()
    x=(age-age.mean())/10
    design=np.column_stack([np.ones(len(x)),x,x*x])
    coefficients,_,rank,_=np.linalg.lstsq(design,y,rcond=1e-10)
    assert rank==3
    return float(100*coefficients[2]/g.seconds.median())


def stint_curves(laps):
    assert not laps.duplicated(['Driver','LapNumber']).any()
    clean=start_exposure(laps)
    field=clean.groupby('LapNumber').seconds
    clean['residual']=clean.seconds-field.transform('median')
    clean=clean[field.transform('count').ge(6)&clean.TyreLife.gt(3)&np.isfinite(clean.TyreLife)]
    rows=[]
    for (driver,stint,compound),g in clean.groupby(['Driver','Stint','Compound'],sort=True):
        if compound not in ('SOFT','MEDIUM','HARD'):
            continue
        value=curvature(g,12)
        if not np.isfinite(value):
            continue
        halves=[g[g.LapNumber.mod(2).eq(h)] for h in (0,1)]
        rows.append(dict(driver=driver,stint=stint,compound=compound,n=len(g),n0=len(halves[0]),n1=len(halves[1]),
            response=value,half0=curvature(halves[0],6),half1=curvature(halves[1],6)))
    return pd.DataFrame(rows,columns=COLS)


def measurements(source,entries):
    assert source.season.eq(2019).all() and entries.season.eq(2019).all()
    keys=['season','round','driver','stint','compound']
    assert not source.duplicated(keys).any() and np.isfinite(source.response).all()
    legal=entries[['season','round','code']].rename(columns={'code':'driver'})
    assert len(source.merge(legal,on=['season','round','driver'],validate='many_to_one'))==len(source)
    columns=['response','half0','half1']
    # Equal stint medians, same-compound field centering, then equal compound means.
    d=source.groupby(['round','driver','compound'],sort=True)[columns].median().reset_index()
    for col in columns:
        field=d.groupby(['round','compound'])[col]
        d[col]=(d[col]-field.transform('mean')).where(field.transform('count').ge(6))
    driver=d.groupby(['round','driver'],sort=True)[columns].mean()
    rows=[]
    for (round_,team),event in entries.groupby(['round','team'],sort=True):
        codes=sorted(event.code.unique()); values=[]
        for col in columns:
            observed=[driver.loc[(round_,code),col] if (round_,code) in driver.index else np.nan for code in codes]
            values.append(float(np.mean(observed)) if len(observed)==2 and np.isfinite(observed).all() else np.nan)
        rows.append(dict(season=2019,round=round_,team=team,response=values[0],half0=values[1],half1=values[2]))
    table=pd.DataFrame(rows)
    for col in columns:
        field=table.groupby('round')[col]
        table['centered_'+col]=(table[col]-field.transform('mean')).where(field.transform('count').ge(6))
    return table


def main():
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True); parts=[]; reports=[]
    for round_ in range(1,22):
        session=cached_session(fastf1,2019,round_)
        current=stint_curves(pd.DataFrame(session.laps)).assign(season=2019,round=round_)
        parts.append(current); reports.append(dict(round=round_,stints=len(current)))
        print('SOURCE',json.dumps(reports[-1]),flush=True)
    source=pd.concat(parts,ignore_index=True)
    source.to_csv(ROOT/'results/e216_stint_curves.csv',index=False)
    entries=load_results().loc[lambda d:d.season.eq(2019),['season','round','code','team']]
    table=measurements(source,entries); result,forecast=evaluate(table)
    errors=pd.read_csv(ROOT/'results/e194_forecasts_errors_scales.csv').loc[lambda d:d.season.eq(2019)]
    errors=errors.groupby(['round','team'],sort=True).error.mean().rename('pace_error').reset_index()
    field=errors.groupby('round').pace_error
    errors['pace_error']=(errors.pace_error-field.transform('mean')).where(field.transform('count').ge(6))
    utility=forecast.merge(errors,on=['round','team'],how='left',validate='one_to_one')
    utility=utility[utility.past_count.ge(3)&utility.pace_error.notna()]
    utility_gate=False
    result.update(utility_rows=len(utility),utility_races=int(utility['round'].nunique()))
    if result['coverage_gate'] and len(utility)>=60 and utility['round'].nunique()>=12:
        correlation,ci=cluster_interval(utility,'prediction','pace_error')
        utility_gate=bool(correlation>=.15 and ci[0]>0)
        result.update(pace_residual_correlation=correlation,pace_residual_ci=ci.tolist())
    result['signature_gate']=result['gate']; result['utility_gate']=utility_gate
    result['gate']=bool(result['signature_gate'] and utility_gate)
    result.update(source=reports,stints=len(source))
    table.to_csv(ROOT/'results/e216_team_measurements.csv',index=False)
    forecast.to_csv(ROOT/'results/e216_temporal_forecasts.csv',index=False)
    utility.to_csv(ROOT/'results/e216_pace_utility.csv',index=False)
    (ROOT/'results/e216_measurement_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
