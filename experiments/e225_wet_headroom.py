"""Cache-only wet-tyre response labels and fixed ensemble-priority diagnostic; no winner fit."""
import json

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.e190_traffic_feasibility import cached_session
from f1.data import ROOT

VALID={'SOFT','MEDIUM','HARD','INTERMEDIATE','WET'}


def wet_label(raw):
    assert not raw.duplicated(['Driver','LapNumber']).any()
    valid=raw.Driver.notna()&raw.LapNumber.ge(1)&raw.Compound.isin(VALID)
    clean=raw[valid]
    support=clean.Driver.nunique()>=10 and float(valid.mean())>=.9
    wet=clean[clean.Compound.isin(['INTERMEDIATE','WET'])].groupby('Driver').size()
    drivers=int(wet.ge(2).sum())
    return dict(available=bool(support),raw_laps=len(raw),valid_laps=int(valid.sum()),
                valid_fraction=float(valid.mean()) if len(raw) else 0.,drivers=int(clean.Driver.nunique()),
                wet_drivers_two_laps=drivers,wet=bool(drivers>=5) if support else None)


def extract():
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True)
    reports=[]; projected=[]
    for path in sorted((ROOT/'results/e223_receipts').glob('*.json')):
        source=json.loads(path.read_text()); plan=source['plan']
        if plan['season']<2019: continue
        assert 2019<=plan['season']<=2022 and source['complete']
        report=dict(season=plan['season'],round=plan['round'],available=False,
                    forecast_rain=source['forecast_precip_mm'])
        try:
            session=cached_session(fastf1,report['season'],report['round'])
            raw=pd.DataFrame(session.laps)[['Driver','LapNumber','Compound']].copy()
            report.update(wet_label(raw))
            projected.append(raw.assign(season=report['season'],round=report['round']))
        except Exception as exc:
            report['error_type']=type(exc).__name__
        reports.append(report)
        print('SOURCE',json.dumps(report),flush=True)
    assert len(reports)==82
    if projected: pd.concat(projected,ignore_index=True).to_csv(ROOT/'results/e225_tyre_records.csv',index=False)
    (ROOT/'results/e225_wet_source.json').write_text(json.dumps(reports,indent=2)+'\n')
    return reports


def evaluate(reports):
    obs=pd.DataFrame(reports).sort_values(['season','round']).reset_index(drop=True)
    usable=obs.available
    coverage={str(y):dict(total=int(obs.season.eq(y).sum()),available=int((obs.season.eq(y)&usable).sum())) for y in range(2019,2023)}
    source_gate=bool(usable.sum()>=78 and all(v['available']/v['total']>=.8 for v in coverage.values()))
    if not source_gate:
        result=dict(source_gate=False,coverage=coverage,gate=False,winner_fit=False)
        (ROOT/'results/e225_wet_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    obs=obs[usable].copy(); obs['climatology']=np.nan
    n=wet=0
    for i,r in obs.iterrows():
        obs.loc[i,'climatology']=(wet+1)/(n+10)
        wet+=int(r.wet); n+=1
    obs['control']=obs.forecast_rain.gt(1).astype(float)
    inner=obs[obs.season.ge(2020)].copy()
    y=inner.wet.to_numpy(float); p=inner.control.to_numpy(float); c=inner.climatology.to_numpy(float)
    difference,ci,tail=W.paired_boot((y-c)**2,(y-p)**2)
    shuffled=np.random.default_rng(225).permutation(y)
    reference=pd.read_csv(ROOT/'results/e224_inner_per_race.csv',float_precision='round_trip')
    # Descriptive frozen-selector headroom, not an executable oracle model.
    diagnostics={}
    for name in ('standings','ens','win_logit'):
        r=reference[reference.model.eq(name)].merge(inner[['season','round','wet']],on=['season','round'],validate='one_to_one')
        assert len(r)==len(inner)
        missed=int((r.wet.astype(bool)&r.winner.eq(0)).sum())
        total=int(reference[reference.model.eq(name)].winner.sum())
        diagnostics[name]=dict(frozen_winners=total,wet_misses_on_observed_inner_races=missed,
                               conditional_wet_only_perfect_correction_hits=total+missed,
                               full_inner_denominator=61,
                               conditional_rate=(total+missed)/61)
    wet_count=int(obs.wet.sum()); mistakes=int((p!=y).sum())
    gate=bool(wet_count>=8 and diagnostics['win_logit']['wet_misses_on_observed_inner_races']>=3 and mistakes>=3)
    result=dict(source_gate=True,coverage=coverage,wet_races=wet_count,available_races=len(obs),
                inner_available=len(inner),inner_wet=int(y.sum()),control_classification_errors=mistakes,
                control_brier=float(np.mean((y-p)**2)),past_climatology_brier=float(np.mean((y-c)**2)),
                control_brier_shuffled_weather_labels=float(np.mean((shuffled-p)**2)),
                climatology_minus_control_brier=dict(difference=float(difference),ci=ci.tolist(),p_nonpositive=float(tail)),
                diagnostics=diagnostics,gate=gate,winner_fit=False)
    obs.to_csv(ROOT/'results/e225_wet_windows.csv',index=False)
    (ROOT/'results/e225_wet_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    print(json.dumps(evaluate(extract()),indent=2),flush=True)


if __name__=='__main__': main()
