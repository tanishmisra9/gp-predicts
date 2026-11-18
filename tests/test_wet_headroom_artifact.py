"""Independent scalar tyre labels, chronology, weather metrics and conditional diagnostic."""
import csv,json,math
from collections import Counter,defaultdict
import numpy as np
import pandas as pd
from f1.data import ROOT


def test_artifact():
    rows=json.loads((ROOT/'results/e225_wet_source.json').read_text())
    result=json.loads((ROOT/'results/e225_wet_diagnostic.json').read_text())
    grouped=defaultdict(list)
    with (ROOT/'results/e225_tyre_records.csv').open() as f:
        for r in csv.DictReader(f): grouped[(int(r['season']),int(r['round']))].append(r)
    assert len(rows)==82 and len(grouped)==82
    labels={}
    for row in rows:
        key=row['season'],row['round']; raw=grouped[key]; valid=[]; seen=set()
        for r in raw:
            lap=float(r['LapNumber']) if r['LapNumber'] else float('nan')
            ident=(r['Driver'],lap if math.isfinite(lap) else None)
            assert ident not in seen; seen.add(ident)
            if r['Driver'] and lap>=1 and r['Compound'] in {'SOFT','MEDIUM','HARD','INTERMEDIATE','WET'}:
                valid.append(r)
        drivers={r['Driver'] for r in valid}; counts=Counter(r['Driver'] for r in valid if r['Compound'] in {'INTERMEDIATE','WET'})
        wetdrivers=sum(n>=2 for n in counts.values()); available=len(drivers)>=10 and len(valid)/len(raw)>=.9
        label=wetdrivers>=5 if available else None
        assert available==row['available'] and len(raw)==row['raw_laps'] and len(valid)==row['valid_laps']
        assert len(drivers)==row['drivers'] and wetdrivers==row['wet_drivers_two_laps'] and label==row['wet']
        assert abs(len(valid)/len(raw)-row['valid_fraction'])<1e-14
        labels[key]=label
    assert sum(v is not None for v in labels.values())==80 and sum(v is True for v in labels.values())==11
    n=wet=0; observed=[]
    for row in sorted(rows,key=lambda r:(r['season'],r['round'])):
        key=row['season'],row['round']; y=labels[key]
        if y is None: continue
        c=(wet+1)/(n+10); p=float(row['forecast_rain']>1)
        if row['season']>=2020: observed.append((key,float(y),c,p))
        wet+=int(y); n+=1
    assert len(observed)==59
    y=np.array([r[1] for r in observed]); c=np.array([r[2] for r in observed]); p=np.array([r[3] for r in observed])
    assert int(sum(y))==result['inner_wet']==10 and int(sum(y!=p))==result['control_classification_errors']==10
    assert abs(float(np.mean((y-p)**2))-result['control_brier'])<1e-14
    assert abs(float(np.mean((y-c)**2))-result['past_climatology_brier'])<1e-14
    diff=(y-c)**2-(y-p)**2
    indices=np.random.default_rng(0).integers(0,len(diff),size=(5000,len(diff)))
    bootstrap=np.array([sum(diff[i] for i in draw)/len(draw) for draw in indices])
    metric=result['climatology_minus_control_brier']
    assert abs(float(diff.mean())-metric['difference'])<1e-14
    np.testing.assert_allclose(np.percentile(bootstrap,[2.5,97.5]),metric['ci'],atol=1e-14,rtol=0)
    assert float(np.mean(bootstrap<=0))==metric['p_nonpositive']
    shuffled=np.random.default_rng(225).permutation(y)
    assert abs(float(np.mean((shuffled-p)**2))-result['control_brier_shuffled_weather_labels'])<1e-14
    reference=pd.read_csv(ROOT/'results/e224_inner_per_race.csv',float_precision='round_trip')
    for name,d in result['diagnostics'].items():
        ref=reference[reference.model.eq(name)]
        hits=int(ref.winner.sum()); assert len(ref)==61
        misses=sum(labels.get((int(r.season),int(r.round))) is True and r.winner==0 for r in ref.itertuples(index=False))
        assert hits==d['frozen_winners'] and misses==d['wet_misses_on_observed_inner_races']
        assert hits+misses==d['conditional_wet_only_perfect_correction_hits']
        assert (hits+misses)/61==d['conditional_rate']
    assert result['source_gate'] and result['gate'] and not result['winner_fit']
    print('All82 scalar tyre labels/59-event weather metrics/past chronology/bootstrap/conditional diagnostics pass')


if __name__=='__main__': test_artifact()
