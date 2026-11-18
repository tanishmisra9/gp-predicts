"""E216 independent compound/team aggregates and forecast/utility statistics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.e191_traffic_repeatability import past_response
from experiments.e216_tyre_curvature import measurements
from tests.test_braking_forecast_artifact import forecast,interval
from f1.data import load_results


def test_artifacts():
    result=json.loads(Path('results/e216_measurement_summary.json').read_text())
    source=pd.read_csv('results/e216_stint_curves.csv',float_precision='round_trip')
    table=pd.read_csv('results/e216_team_measurements.csv',float_precision='round_trip')
    saved=pd.read_csv('results/e216_temporal_forecasts.csv',float_precision='round_trip')
    entries=load_results().loc[lambda d:d.season.eq(2019),['season','round','code','team']]
    cols=['response','half0','half1']
    driver_compound={}
    for (round_,driver,compound),g in source.groupby(['round','driver','compound'],sort=True):
        driver_compound[(round_,driver,compound)]=[float(np.median(g[c].dropna())) if g[c].notna().any() else np.nan for c in cols]
    for round_ in range(1,22):
        for compound in ('SOFT','MEDIUM','HARD'):
            keys=[k for k in driver_compound if k[0]==round_ and k[2]==compound]
            for j in range(3):
                valid=[driver_compound[k][j] for k in keys if np.isfinite(driver_compound[k][j])]
                center=float(np.mean(valid)) if len(valid)>=6 else np.nan
                for k in keys:
                    driver_compound[k][j]-=center
    driver={}
    for row in entries.itertuples():
        values=[]
        for j in range(3):
            v=[a[j] for (r,d,c),a in driver_compound.items() if r==row.round and d==row.code and np.isfinite(a[j])]
            values.append(float(np.mean(v)) if v else np.nan)
        driver[(row.round,row.code)]=values
    rows=[]
    for (round_,team),event in entries.groupby(['round','team'],sort=True):
        values=[]
        for j in range(3):
            v=[driver[(round_,d)][j] for d in sorted(event.code.unique())]
            values.append(float(np.mean(v)) if len(v)==2 and np.isfinite(v).all() else np.nan)
        rows.append(dict(season=2019,round=round_,team=team,response=values[0],half0=values[1],half1=values[2]))
    expected=pd.DataFrame(rows)
    for col in cols:
        expected['centered_'+col]=np.nan
        for _,ids in expected.groupby('round').groups.items():
            v=expected.loc[ids,col]; good=v.notna()
            if good.sum()>=6:
                expected.loc[ids,'centered_'+col]=v-v[good].mean()
    pd.testing.assert_frame_equal(table,expected,check_dtype=False,atol=1e-12,rtol=1e-12)
    predicted,count=forecast(expected)
    np.testing.assert_allclose(saved.prediction,predicted,atol=1e-12)
    np.testing.assert_array_equal(saved.past_count,count)
    halves=expected.dropna(subset=['centered_half0','centered_half1'])
    eligible=expected[expected.centered_response.notna()&(count>=3)].copy()
    eligible['prediction']=predicted[eligible.index]
    races=sum(g.centered_response.notna().sum()>=6 for _,g in expected.groupby('round'))
    coverage=bool(races>=12 and len(halves)>=60 and len(eligible)>=60)
    assert [races,len(halves),len(eligible)]==[result['valid_races'],result['split_rows'],result['temporal_rows']]
    assert coverage==result['coverage_gate']
    if coverage:
        split,split_ci=interval(halves,'centered_half0','centered_half1')
        temporal,temporal_ci=interval(eligible,'prediction','centered_response')
        eligible['mse_gain']=eligible.centered_response**2-(eligible.prediction-eligible.centered_response)**2
        gain,gain_ci=interval(eligible,'mse_gain')
        null=expected.copy(); rng=np.random.default_rng(204)
        for _,ids in null[null.centered_response.notna()].groupby('round').groups.items():
            null.loc[ids,'centered_response']=null.loc[rng.permutation(ids),'centered_response'].to_numpy()
        null_pred,_=forecast(null)
        eligible['null_gain']=(null_pred[eligible.index]-eligible.centered_response)**2-(eligible.prediction-eligible.centered_response)**2
        null_gain,null_ci=interval(eligible,'null_gain')
        for name,value,ci in [('split_correlation',split,split_ci),('temporal_correlation',temporal,temporal_ci),('mse_gain',gain,gain_ci),('null_mse_gain',null_gain,null_ci)]:
            np.testing.assert_allclose(result[name],value,atol=1e-10)
            key={'split_correlation':'split_ci','temporal_correlation':'temporal_ci','mse_gain':'mse_gain_ci','null_mse_gain':'null_mse_gain_ci'}[name]
            np.testing.assert_allclose(result[key],ci,atol=1e-10)
        assert result['signature_gate']==bool(split>=.3 and split_ci[0]>0 and temporal>=.15 and gain>0 and gain_ci[0]>0 and null_gain>0 and null_ci[0]>0)
    else:
        assert not result['signature_gate'] and 'split_correlation' not in result
    errors=pd.read_csv('results/e194_forecasts_errors_scales.csv').loc[lambda d:d.season.eq(2019)].copy()
    for _,g in errors.groupby('round',sort=True):
        valid=g.y_pace.notna()
        delta=(g.loc[valid,'y_pace']-g.loc[valid,'mean_forecast']).to_numpy()
        np.testing.assert_allclose(g.loc[valid,'error'],delta-np.mean(delta),atol=1e-12)
    groups=[]
    for (round_,team),g in errors.groupby(['round','team'],sort=True):
        v=g.error.dropna().to_numpy()
        groups.append(dict(round=round_,team=team,pace_error=float(np.mean(v)) if len(v) else np.nan))
    targets=pd.DataFrame(groups)
    for _,ids in targets.groupby('round').groups.items():
        v=targets.loc[ids,'pace_error']; good=v.notna()
        targets.loc[ids,'pace_error']=v-v[good].mean() if good.sum()>=6 else np.nan
    utility=saved.merge(targets,on=['round','team'],how='left',validate='one_to_one')
    utility=utility[utility.past_count.ge(3)&utility.pace_error.notna()]
    observed=pd.read_csv('results/e216_pace_utility.csv',float_precision='round_trip')
    pd.testing.assert_frame_equal(utility.reset_index(drop=True),observed,check_dtype=False,atol=1e-12,rtol=1e-12)
    assert result['utility_rows']==len(utility) and result['utility_races']==utility['round'].nunique()
    utility_gate=False
    if coverage and len(utility)>=60 and utility['round'].nunique()>=12:
        corr,ci=interval(utility,'prediction','pace_error')
        np.testing.assert_allclose(result['pace_residual_correlation'],corr,atol=1e-10)
        np.testing.assert_allclose(result['pace_residual_ci'],ci,atol=1e-10)
        utility_gate=bool(corr>=.15 and ci[0]>0)
    assert result['utility_gate']==utility_gate and result['gate']==bool(result['signature_gate'] and utility_gate)
    changed=source.copy(); changed.loc[changed['round'].ge(10),'response']=2.9
    again=past_response(measurements(changed,entries))
    np.testing.assert_array_equal(saved.loc[saved['round'].le(10),'prediction'],again.loc[again['round'].le(10),'prediction'])
    prefix=past_response(measurements(source[source['round'].lt(10)],entries))
    np.testing.assert_array_equal(saved.loc[saved['round'].eq(10),'prediction'],prefix.loc[prefix['round'].eq(10),'prediction'])
    earlier=source.copy(); observation=table[table['round'].lt(10)&table.centered_response.notna()].iloc[0]
    codes=entries[entries['round'].eq(observation['round'])&entries.team.eq(observation.team)].code
    earlier.loc[earlier['round'].eq(observation['round'])&earlier.driver.isin(codes),'response']+=.5
    prior=past_response(measurements(earlier,entries))
    assert prior.loc[prior['round'].eq(10),'prediction'].ne(saved.loc[saved['round'].eq(10),'prediction']).any()


if __name__=='__main__':
    test_artifacts()
    print('E216 independent lap/team aggregates, scalar histories, null/bootstrap/pace utility/gates and source timing PASS')
