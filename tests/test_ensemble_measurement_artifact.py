"""Independent fixed59-label Brier/resampling/null/fallback/gate audit, no winner fit."""
import csv,json
from pathlib import Path
import numpy as np


def test_measurement():
    root=Path(__file__).resolve().parents[1]
    result=json.loads((root/'results/e226_measurement_summary.json').read_text())
    source=json.loads((root/'results/e226_source_summary.json').read_text())
    assert result['source_gate']==source['source_gate'] and not result['winner_fit']
    if not source['source_gate']:
        assert not result['gate'];return
    with (root/'results/e226_ensemble_windows.csv').open() as f:
        forecasts={(int(r['season']),int(r['round'])):r for r in csv.DictReader(f)}
    with (root/'results/e225_wet_windows.csv').open() as f:
        observed=[r for r in csv.DictReader(f) if int(r['season'])>=2020]
    assert len(observed)==59 and len(forecasts)==82
    ys=[];ps=[];controls=[];climates=[];fallback=0
    for r in observed:
        key=int(r['season']),int(r['round']);f=forecasts[key]
        y=float(r['wet']=='True');p=float(f['p_wet']);base=float(r['control'])
        assert base==float(f['control']) and 0<=p<=1
        if f['complete']=='False':assert p==base;fallback+=1
        ys.append(y);ps.append(p);controls.append(base);climates.append(float(r['climatology']))
    y=np.array(ys);p=np.array(ps);error=(y-p)**2
    assert result['inner_supported']==59 and result['inner_wet']==int(y.sum()) and fallback==result['fallback_inner']
    assert abs(float(sum(error)/59)-result['brier'])<1e-14
    diffs={}
    for name,baseline in [('control',controls),('climatology',climates)]:
        prior=(y-np.array(baseline))**2;diff=prior-error
        indices=np.random.default_rng(0).integers(0,59,size=(5000,59))
        samples=np.array([sum(diff[i] for i in draw)/59 for draw in indices])
        metric=result['paired'][name]
        assert abs(float(sum(prior)/59)-metric['brier'])<1e-14
        assert abs(float(sum(diff)/59)-metric['difference'])<1e-14
        np.testing.assert_allclose(np.percentile(samples,[2.5,97.5]),metric['ci'],rtol=0,atol=1e-14)
        assert float(np.mean(samples<=0))==metric['p_nonpositive']
        diffs[name]=(float(sum(diff)/59),float(np.percentile(samples,2.5)))
    shuffled=np.random.default_rng(226).permutation(y)
    null=float(sum((shuffled-p)**2)/59)
    assert abs(null-result['shuffled_weather_labels_brier'])<1e-14
    gate=all(d>=.01 and low>0 for d,low in diffs.values()) and float(error.mean())<null
    assert gate==result['gate']
    print('Independent59-event Brier/baselines/bootstrap/null/fallback/gate audit passes; gate',gate)


if __name__=='__main__':test_measurement()
