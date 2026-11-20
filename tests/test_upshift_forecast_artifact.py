"""E244 independently reconstruct team measurements, forecasts and race CIs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.e191_traffic_repeatability import past_response
from experiments.e211_braking_signature import team_table
from f1.data import load_results


def forecast(frame):
    result=[]; counts=[]
    for r in frame.itertuples():
        numerator=0.; denominator=5.; count=0
        for p in frame.itertuples():
            if p.team==r.team and p.round<r.round and pd.notna(p.centered_response):
                weight=2**(-(r.round-p.round)/10)
                numerator+=weight*p.centered_response; denominator+=weight; count+=1
        result.append(numerator/denominator); counts.append(count)
    return np.array(result),np.array(counts)


def interval(frame,x,y=None):
    groups=[]
    for _,g in frame.groupby('round',sort=True):
        a=g[x].to_numpy(float)
        if y is None:
            groups.append([len(a),a.sum()])
        else:
            b=g[y].to_numpy(float)
            groups.append([len(a),a.sum(),b.sum(),a@a,b@b,a@b])
    groups=np.array(groups)
    indices=np.random.default_rng(191).integers(len(groups),size=(4000,len(groups)))
    samples=groups[indices].sum(axis=1)
    def statistic(total):
        if y is None:
            return total[...,1]/total[...,0]
        n,a,b,aa,bb,ab=np.moveaxis(total,-1,0)
        return (ab-a*b/n)/np.sqrt((aa-a*a/n)*(bb-b*b/n))
    distribution=statistic(samples); distribution=distribution[np.isfinite(distribution)]
    return float(statistic(groups.sum(axis=0))),np.quantile(distribution,[.025,.975])


def test_artifacts():
    result=json.loads(Path('results/e244_measurement_summary.json').read_text())
    source=pd.read_csv('results/e244_upshift_events.csv',float_precision='round_trip')
    table=pd.read_csv('results/e244_team_measurements.csv',float_precision='round_trip')
    saved=pd.read_csv('results/e244_temporal_forecasts.csv',float_precision='round_trip')
    entries=load_results().loc[lambda d:d.season.eq(2019),['season','round','code','team']]
    from collections import defaultdict
    from statistics import median
    raw_laps=defaultdict(list)
    for r in source.itertuples():
        raw_laps[(r.round,r.driver,r.lap)].append(r.duration)
    laps={key:median(values) for key,values in raw_laps.items()}
    rows=[]
    for (round_,team),event in entries.groupby(['round','team'],sort=True):
        values=[]
        for subset in ('full',0,1):
            drivers=[]
            for code in sorted(event.code.unique()):
                v=[value for (r,d,lap),value in laps.items() if r==round_ and d==code and (subset=='full' or lap%2==subset)]
                drivers.append(float(np.mean(v)) if len(v)>=(10 if subset=='full' else 5) else np.nan)
            values.append(float(np.mean(drivers)) if len(drivers)==2 and np.isfinite(drivers).all() else np.nan)
        rows.append(dict(season=2019,round=round_,team=team,response=values[0],half0=values[1],half1=values[2]))
    expected=pd.DataFrame(rows)
    for col in ('response','half0','half1'):
        expected['centered_'+col]=np.nan
        for _,ids in expected.groupby('round').groups.items():
            values=expected.loc[ids,col]; good=values.notna()
            if good.sum()>=6:
                expected.loc[ids,'centered_'+col]=values-values[good].mean()
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
        assert result['gate']==bool(split>=.3 and split_ci[0]>0 and temporal>=.15 and gain>0 and gain_ci[0]>0 and null_gain>0 and null_ci[0]>0)
    else:
        assert not result['gate'] and 'split_correlation' not in result
    changed=source.copy(); changed.loc[changed['round'].ge(10),'duration']=14.9
    again=past_response(team_table(changed,entries))
    np.testing.assert_array_equal(saved.loc[saved['round'].le(10),'prediction'],again.loc[again['round'].le(10),'prediction'])
    prefix=past_response(team_table(source[source['round'].lt(10)],entries))
    np.testing.assert_array_equal(saved.loc[saved['round'].eq(10),'prediction'],prefix.loc[prefix['round'].eq(10),'prediction'])
    earlier=source.copy(); observation=table[table['round'].lt(10)&table.centered_response.notna()].iloc[0]
    codes=entries[entries['round'].eq(observation['round'])&entries.team.eq(observation.team)].code
    earlier.loc[earlier['round'].eq(observation['round'])&earlier.driver.isin(codes),'duration']+=.5
    prior=past_response(team_table(earlier,entries))
    assert prior.loc[prior['round'].eq(10),'prediction'].ne(saved.loc[saved['round'].eq(10),'prediction']).any()


if __name__=="__main__":
    test_artifacts()
    print("E244 independent lap/team/parity/scalar histories/null/bootstrap/gate and real source timing PASS")
