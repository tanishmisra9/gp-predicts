"""Independent E234 scalar history, race bootstrap/null and source timing audit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from experiments.e234_launch_transfer import history
from tests.test_braking_forecast_artifact import interval


def scalar(frame):
    predictions=[]; counts=[]
    for row in frame.itertuples():
        past=[p for p in frame.itertuples() if p.driver==row.driver and p.round<row.round and pd.notna(p.response)]
        weights=[2**(-(row.round-p.round)/10) for p in past]
        predictions.append(sum(w*p.response for w,p in zip(weights,past))/(5+sum(weights)))
        counts.append(len(past))
    return np.asarray(predictions),np.asarray(counts)


def test_artifacts():
    reports=json.loads(Path('results/e234_launch_source.json').read_text())
    result=json.loads(Path('results/e234_transfer_summary.json').read_text())
    source=pd.read_csv('results/e234_driver_measurements.csv',float_precision='round_trip')
    saved=pd.read_csv('results/e234_temporal_forecasts.csv',float_precision='round_trip')
    rows=[]; widths=[]; pairs=0; robust=0
    for report in reports['sessions']:
        good=[r for r in report['rows'] if r['supported']] if report['gate'] else []
        for r in report['rows']:
            rows.append(dict(season=2019,round=report['round'],driver=r['driver'],response=r.get('relative20',np.nan) if report['gate'] else np.nan))
        for i,a in enumerate(good):
            widths.append(a['bracket20_high']-a['bracket20_low'])
            for b in good[i+1:]:
                pairs+=1
                robust+=a['bracket20_high']<b['bracket20_low'] or b['bracket20_high']<a['bracket20_low']
    pd.testing.assert_frame_equal(source,pd.DataFrame(rows),atol=1e-12,rtol=1e-12)
    assert result['pairs']==pairs and result['robust_pairs']==robust
    np.testing.assert_allclose(result['mean_bracket_width'],np.mean(widths),atol=1e-12)
    np.testing.assert_allclose(result['robust_pair_fraction'],robust/pairs,atol=1e-12)
    count=sum(r['gate'] for r in reports['sessions']); measured=int(source.response.notna().sum())
    assert (result['valid_events'],result['measured_rows'])==(count,measured)
    source_gate=count>=16 and measured>=300
    assert result['source_gate']==source_gate==reports['gate']
    prediction,counts=scalar(source)
    np.testing.assert_allclose(saved.prediction,prediction,atol=1e-14,rtol=1e-12)
    np.testing.assert_array_equal(saved.past_count,counts)
    eligible=source[(counts>=3)&source.response.notna()].copy()
    eligible['prediction']=prediction[eligible.index]
    coverage=bool(source_gate and len(eligible)>=120 and eligible['round'].nunique()>=12)
    assert result['coverage_gate']==coverage and result['temporal_rows']==len(eligible) and result['temporal_races']==eligible['round'].nunique()
    gate=False
    if coverage:
        corr,ci=interval(eligible,'prediction','response')
        eligible['mse_gain']=eligible.response**2-(eligible.prediction-eligible.response)**2
        gain,gain_ci=interval(eligible,'mse_gain')
        null=source.copy(); rng=np.random.default_rng(234)
        for _,ids in null[null.response.notna()].groupby('round').groups.items():
            null.loc[ids,'response']=null.loc[rng.permutation(ids),'response'].to_numpy()
        shuffled,_=scalar(null)
        eligible['null_gain']=(shuffled[eligible.index]-eligible.response)**2-(eligible.prediction-eligible.response)**2
        ng,nci=interval(eligible,'null_gain')
        for key,value in [('correlation',corr),('correlation_ci',ci),('mse_gain',gain),('mse_gain_ci',gain_ci),('null_mse_gain',ng),('null_mse_gain_ci',nci),('baseline_mse',(eligible.response**2).mean()),('forecast_mse',((eligible.prediction-eligible.response)**2).mean())]:
            np.testing.assert_allclose(result[key],value,atol=1e-10,rtol=1e-10)
        gate=bool(corr>=.15 and ci[0]>0 and gain>0 and gain_ci[0]>0 and ng>0 and nci[0]>0)
    assert result['gate']==gate and not result['winner_fit']
    altered=source.copy(); altered.loc[altered['round'].ge(10),'response']=100
    changed=history(altered)
    np.testing.assert_array_equal(saved.loc[saved['round'].le(10),'prediction'],changed.loc[changed['round'].le(10),'prediction'])
    prefix=source.copy(); prefix.loc[prefix['round'].ge(10),'response']=np.nan
    changed=history(prefix)
    np.testing.assert_array_equal(saved.loc[saved['round'].le(10),'prediction'],changed.loc[changed['round'].le(10),'prediction'])
    earlier=source.copy(); first=earlier[earlier['round'].lt(10)&earlier.response.notna()].index[0]
    earlier.loc[first,'response']+=1
    changed=history(earlier)
    assert changed.loc[changed['round'].eq(10),'prediction'].ne(saved.loc[saved['round'].eq(10),'prediction']).any()
    old=json.loads(Path('results/e233_launch_source.json').read_text())
    for r in old['sessions']:
        current=next(x for x in reports['sessions'] if x['round']==r['round'])
        for key,value in r.items():
            if key!='rows': assert current[key]==value
            else:
                for new,prior in zip(current[key],value):
                    assert {k:v for k,v in new.items() if not k.startswith('bracket20_')}==prior


if __name__=='__main__':
    test_artifacts()
    print('E234 independent driver histories/zero/null/bootstrap/sampling/source-timing/gates PASS')
