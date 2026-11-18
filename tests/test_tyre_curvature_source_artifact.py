"""Independent E216 raw filters and quadratic moment-equation fits."""
import json
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

import f1.ff1
from experiments.e190_traffic_feasibility import cached_session


def test_source():
    saved=pd.read_csv('results/e216_stint_curves.csv',float_precision='round_trip')
    result=json.loads(Path('results/e216_measurement_summary.json').read_text())
    fastf1.Cache.offline_mode(True); all_rows=[]
    for round_ in range(1,22):
        raw=pd.DataFrame(cached_session(fastf1,2019,round_).laps)
        assert not raw.duplicated(['Driver','LapNumber']).any()
        keep=raw.LapNumber.gt(1)&raw.LapTime.notna()&raw.Time.notna()&raw.LapStartTime.notna()
        keep&=raw.PitInTime.isna()&raw.PitOutTime.isna()&raw.TrackStatus.astype(str).eq('1')&raw.IsAccurate.fillna(False)
        clean=raw[keep].copy(); clean['seconds']=clean.LapTime.dt.total_seconds()
        medians={d:np.median(g.seconds) for d,g in clean.groupby('Driver')}
        clean=clean[[r.seconds<1.07*medians[r.Driver] for r in clean.itertuples()]]
        field={n:(len(g),float(np.median(g.seconds))) for n,g in clean.groupby('LapNumber')}
        clean['residual']=[r.seconds-field[r.LapNumber][1] for r in clean.itertuples()]
        clean=clean[[field[r.LapNumber][0]>=6 and np.isfinite(r.TyreLife) and r.TyreLife>3 for r in clean.itertuples()]]
        def moment_fit(g,minimum):
            ages=list(g.TyreLife)
            if len(g)<minimum or len(set(ages))!=len(ages) or max(ages)-min(ages)<10:
                return np.nan
            middle=sum(ages)/len(ages); x=[(a-middle)/10 for a in ages]
            matrix=np.array([[sum(a**(i+j) for a in x) for j in range(3)] for i in range(3)])
            vector=np.array([sum(a**i*y for a,y in zip(x,g.residual)) for i in range(3)])
            coefficients=np.linalg.solve(matrix,vector)
            return 100*coefficients[2]/np.median(g.seconds)
        rows=[]
        for (driver,stint,compound),g in clean.groupby(['Driver','Stint','Compound'],sort=True):
            if compound not in ('SOFT','MEDIUM','HARD'):
                continue
            value=moment_fit(g,12)
            if not np.isfinite(value):
                continue
            even=g[g.LapNumber.mod(2).eq(0)]; odd=g[g.LapNumber.mod(2).eq(1)]
            rows.append(dict(driver=driver,stint=stint,compound=compound,n=len(g),n0=len(even),n1=len(odd),
                response=value,half0=moment_fit(even,6),half1=moment_fit(odd,6),season=2019,round=round_))
        assert result['source'][round_-1]==dict(round=round_,stints=len(rows))
        all_rows.extend(rows)
    expected=pd.DataFrame(all_rows); keys=['season','round','driver','stint','compound']
    pd.testing.assert_frame_equal(saved.sort_values(keys).reset_index(drop=True),expected[saved.columns].sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-10,rtol=1e-10)
    assert result['stints']==len(saved)==len(expected)


if __name__=='__main__':
    test_source()
    print('E216 independent all21raw filters/field medians/age support/full and parity moment fits PASS')
