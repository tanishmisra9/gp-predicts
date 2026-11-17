"""Rebuild E204 from raw laps and a separate linear extrapolation."""
import json
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

import f1.ff1
from f1.data import load_results
from experiments.e190_traffic_feasibility import cached_session


def test_artifacts():
    report=json.loads(Path('results/e204_warmup_summary.json').read_text())
    saved=pd.read_csv('results/e204_warmup_blocks.csv',float_precision='round_trip')
    table=pd.read_csv('results/e204_warmup_measurements.csv',float_precision='round_trip')
    forecast=pd.read_csv('results/e204_warmup_forecasts.csv',float_precision='round_trip')
    entries=load_results().query('season==2019')[['round','code','team']]
    expected=[]; status=[]; fastf1.Cache.offline_mode(True)
    for round_ in range(1,22):
        raw=pd.DataFrame(cached_session(fastf1,2019,round_).laps)
        mask=(raw.LapNumber>1)&raw.LapTime.notna()&raw.Time.notna()&raw.LapStartTime.notna()
        mask&=raw.PitInTime.isna()&raw.PitOutTime.isna()&raw.TrackStatus.astype(str).eq('1')&raw.IsAccurate.fillna(False)
        clean=raw[mask].copy(); clean['s']=clean.LapTime.dt.total_seconds()
        medians={d:np.median(g.s) for d,g in clean.groupby('Driver')}
        clean=clean[[r.s<1.07*medians[r.Driver] for r in clean.itertuples()]]
        count=0
        for driver,driver_rows in raw.groupby('Driver'):
            for out in driver_rows[driver_rows.PitOutTime.notna()].itertuples():
                prior=driver_rows[driver_rows.LapNumber.eq(out.LapNumber-1)]
                if len(prior)!=1 or pd.isna(prior.PitInTime.iloc[0]):
                    continue
                gap=(out.PitOutTime-prior.PitInTime.iloc[0]).total_seconds()
                if not 10<gap<60 or out.Compound not in ('SOFT','MEDIUM','HARD'):
                    continue
                group=clean[clean.Driver.eq(driver)&clean.LapNumber.between(out.LapNumber+1,out.LapNumber+4)].sort_values('LapNumber')
                if len(group)!=4 or not np.array_equal(group.LapNumber,np.arange(out.LapNumber+1,out.LapNumber+5)):
                    continue
                if not group.Stint.eq(out.Stint).all() or not group.Compound.eq(out.Compound).all():
                    continue
                if not group.Position.notna().all() or len(set(group.Position))!=1 or not np.allclose(np.diff(group.TyreLife),1):
                    continue
                times=group.s.to_numpy(); _,intercept=np.polyfit([1,2,3],times[1:],1)
                value=(times[0]-intercept)/np.mean(times)*100
                match=saved[saved['round'].eq(round_)&saved.driver.eq(driver)&saved.pitout_lap.eq(out.LapNumber)]
                assert len(match)==1; row=match.iloc[0]
                assert row.first_full_lap==out.LapNumber+1 and row.stint==out.Stint and row.compound==out.Compound
                np.testing.assert_allclose(row[[f'seconds{i}' for i in range(4)]].to_numpy(float),times,atol=1e-12)
                np.testing.assert_allclose(row.contrast,value,atol=1e-10)
                team=entries[entries['round'].eq(round_)&entries.code.eq(driver)].team.item()
                assert row.team==team and row.season==2019
                expected.append(dict(round=round_,team=team,driver=driver,contrast=value)); count+=1
        status.append(dict(round=round_,blocks=count))
    assert report['source']==status and report['blocks']==len(expected)==len(saved)==244
    assert not saved.duplicated(['round','driver','pitout_lap']).any()
    values=pd.DataFrame(expected); reconstructed=[]
    for (round_,team),event in entries.groupby(['round','team'],sort=True):
        drivers=sorted(event.code); group=values[values['round'].eq(round_)&values.team.eq(team)]
        a=[group[group.driver.eq(d)].contrast.median() for d in drivers]
        full=np.mean(a) if len(a)==2 and np.isfinite(a).all() else np.nan
        reconstructed.append(dict(round=round_,team=team,blocks=len(group),half0=a[0],half1=a[1],response=full))
    built=pd.DataFrame(reconstructed)
    for col in ('response','half0','half1'):
        built['centered_'+col]=np.nan
        for _,ids in built.groupby('round').groups.items():
            g=built.loc[ids,col]; good=g.notna()
            if good.sum()>=6:
                built.loc[ids,'centered_'+col]=g-g[good].mean()
    pd.testing.assert_frame_equal(table.drop(columns='season'),built,check_dtype=False,atol=1e-10,rtol=1e-10)
    for i,row in built.iterrows():
        past=built[built['round'].lt(row['round'])&built.team.eq(row.team)&built.centered_response.notna()]
        w=np.array([.5**((row['round']-r)/10) for r in past['round']])
        prediction=np.dot(w,past.centered_response)/(5+w.sum())
        assert forecast.past_count.iloc[i]==len(past)
        np.testing.assert_allclose(forecast.prediction.iloc[i],prediction,atol=1e-10)
    races=int(built.groupby('round').centered_response.count().ge(6).sum())
    halves=int((built.centered_half0.notna()&built.centered_half1.notna()).sum())
    temporal=int((forecast.past_count.ge(3)&built.centered_response.notna()).sum())
    assert [races,halves,temporal]==[report['valid_races'],report['split_rows'],report['temporal_rows']]==[2,22,0]
    assert not report['coverage_gate'] and not report['gate'] and 'split_correlation' not in report


if __name__=='__main__':
    test_artifacts()
    print('E204 independent raw blocks, regression contrasts, team centers, prior forecasts and gate PASS')
