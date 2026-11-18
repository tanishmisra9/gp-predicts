"""Frozen wet-tyre proxy Brier test; no winner fit or source selection."""
import json
import numpy as np
import pandas as pd
import f1.walkforward as W
from f1.data import ROOT


def main():
    source=json.loads((ROOT/'results/e226_source_summary.json').read_text())
    if not source['source_gate']:
        result=dict(source_gate=False,gate=False,winner_fit=False)
    else:
        forecast=pd.read_csv(ROOT/'results/e226_ensemble_windows.csv',float_precision='round_trip')
        observed=pd.read_csv(ROOT/'results/e225_wet_windows.csv',float_precision='round_trip')
        obs=observed[observed.season.ge(2020)].merge(forecast,on=['season','round'],validate='one_to_one',suffixes=('','_new'))
        assert len(obs)==59 and np.array_equal(obs.control,obs.control_new)
        y=obs.wet.to_numpy(float); p=obs.p_wet.to_numpy(float)
        error=(y-p)**2; paired={}
        for name in ('control','climatology'):
            baseline=(y-obs[name].to_numpy(float))**2
            diff,ci,tail=W.paired_boot(baseline,error)
            paired[name]=dict(difference=float(diff),ci=ci.tolist(),p_nonpositive=float(tail),brier=float(baseline.mean()))
        shuffled=np.random.default_rng(226).permutation(y)
        null=float(np.mean((shuffled-p)**2))
        gate=all(v['difference']>=.01 and v['ci'][0]>0 for v in paired.values()) and float(error.mean())<null
        result=dict(source_gate=True,gate=bool(gate),inner_supported=59,inner_wet=int(y.sum()),
                    fallback_inner=int((~obs.complete).sum()),brier=float(error.mean()),
                    shuffled_weather_labels_brier=null,paired=paired,winner_fit=False)
        obs.to_csv(ROOT/'results/e226_wet_measurement.csv',index=False)
    (ROOT/'results/e226_measurement_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
