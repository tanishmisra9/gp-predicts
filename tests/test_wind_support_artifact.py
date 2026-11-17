"""E203 independently reconstruct wind joins, nuisance columns and SVD support."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.e203_wind_support import measure


def test_artifacts():
    raw=pd.read_csv('results/e197_weather_aligned_laps.csv',float_precision='round_trip')
    weather=pd.read_csv('results/e197_weather_samples.csv',float_precision='round_trip')
    assert not weather.duplicated(['season','round','Time']).any()
    lookup={(int(r.season),int(r.round),pd.Timedelta(r.Time)):r.WindSpeed for r in weather.itertuples()}
    raw['WindSpeed']=[lookup[(int(r.season),int(r.round),pd.Timedelta(r.weather_time))] for r in raw.itertuples()]
    assert np.isfinite(raw.WindSpeed).all() and raw.WindSpeed.ge(0).all()
    report=json.loads(Path('results/e203_wind_support_summary.json').read_text())
    diagnostics=pd.read_csv('results/e203_wind_support.csv',float_precision='round_trip')
    residuals=pd.read_csv('results/e203_wind_residuals.csv',float_precision='round_trip')
    assert report['source_rows']==report['eligible_rows']==3151 and report['excluded_rows']==0
    assert len(diagnostics)==180 and len(residuals)==2*len(raw)
    assert not diagnostics.duplicated(['season','round','driver','subset']).any()
    assert not residuals.duplicated(['season','round','Driver','LapNumber','subset']).any()
    assert raw.Rainfall.eq(False).all() and raw[['TrackTemp','TyreLife','Position','Stint','Compound']].notna().all().all()
    assert np.isfinite(raw[['TrackTemp','TyreLife','Position']]).all().all()
    assert raw.close.isin([True,False]).all()
    raw['LapStartTime']=pd.to_timedelta(raw.LapStartTime)
    raw['weather_time']=pd.to_timedelta(raw.weather_time)
    assert raw.LapStartTime.notna().all() and raw.weather_time.notna().all()
    residuals['weather_time']=pd.to_timedelta(residuals.weather_time)
    expected_sessions=[]
    for (season,round_),event in raw.groupby(['season','round'],sort=True):
        assert season==2019 and round_ in (5,12,14)
        start=event.LapStartTime.dt.total_seconds().to_numpy()
        event=event.copy(); event['t']=(start-start.min())/(start.max()-start.min())
        _,inverse=np.unique(event.weather_time.to_numpy(),return_inverse=True)
        event['block_half']=inverse%2
        passing=0
        for driver,all_rows in event.groupby('Driver',sort=True):
            tests=[]
            for name,g in [('full',all_rows),('half0',all_rows[all_rows.block_half.eq(0)]),('half1',all_rows[all_rows.block_half.eq(1)])]:
                t=g.t.to_numpy(); age=g.TyreLife.to_numpy()/10
                cols=[np.ones(len(g)),t,t*t,t*t*t,g.Position.to_numpy()/20,g.close.to_numpy(float)]
                for stint in sorted(g.Stint.unique())[1:]:
                    cols.append((g.Stint.to_numpy()==stint).astype(float))
                for compound in sorted(g.Compound.unique()):
                    selected=(g.Compound.to_numpy()==compound).astype(float)
                    cols.extend([selected*age,selected*age*age])
                cols.append(g.TrackTemp.to_numpy()/10)
                Z=np.column_stack(cols); wind=g.WindSpeed.to_numpy()
                # Project onto an orthonormal basis, without fitting regression coefficients.
                U,s,_=np.linalg.svd(Z,full_matrices=False)
                rank=int((s>s[0]*1e-10).sum()); basis=U[:,:rank]
                expected=wind-basis@(basis.T@wind)
                np.testing.assert_allclose(Z.T@expected,0.,atol=1e-8)
                rms=float(np.sqrt(np.mean(expected*expected)))
                fraction=float(np.mean(expected*expected)/np.var(wind))
                dof=len(g)-rank; readings=g.weather_time.nunique()
                supported=bool(len(g)>=15 and readings>=10 and dof>=10 and rms>=.5 and fraction>=.1)
                saved=diagnostics[diagnostics.season.eq(season)&diagnostics['round'].eq(round_)&diagnostics.driver.eq(driver)&diagnostics.subset.eq(name)].iloc[0]
                assert saved['rows']==len(g) and saved['readings']==readings and saved['rank']==rank and saved['dof']==dof
                np.testing.assert_allclose([saved['residual_rms'],saved['variance_fraction']],[rms,fraction],atol=1e-9)
                assert saved['support']==supported
                saved_rows=residuals[residuals.season.eq(season)&residuals['round'].eq(round_)&residuals.Driver.eq(driver)&residuals.subset.eq(name)].set_index('LapNumber').loc[g.LapNumber]
                assert len(saved_rows)==len(g)
                np.testing.assert_allclose(saved_rows.residual_wind,expected,atol=1e-9)
                np.testing.assert_array_equal(saved_rows.weather_time,g.weather_time)
                np.testing.assert_array_equal(saved_rows.half,g.block_half)
                np.testing.assert_allclose(saved_rows.WindSpeed,wind,atol=1e-12)
                np.testing.assert_allclose(saved_rows.phase,g.t,atol=1e-12)
                tests.append(supported)
            passing+=int(all(tests))
        expected_sessions.append(dict(season=int(season),round=int(round_),drivers=int(event.Driver.nunique()),passing_drivers=passing))
    assert report['sessions']==expected_sessions
    assert report['gate']==(sum(r['passing_drivers']>=10 for r in expected_sessions)>=2)
    # There is no response label in the design: corrupting lap times/outcomes cannot alter it.
    changed=raw.copy()
    changed['seconds']=np.random.default_rng(198).normal(size=len(changed))
    changed['order']=999
    same,table,rows=measure(changed)
    assert same==report
    np.testing.assert_allclose(table[['residual_rms','variance_fraction']],diagnostics[['residual_rms','variance_fraction']],atol=1e-12)
    np.testing.assert_allclose(rows.residual_wind,residuals.residual_wind,atol=1e-12)


if __name__=='__main__':
    test_artifacts()
    print('E203 independent wind merge/design/SVD, shared split, diagnostics/gate and response-label invariance PASS')
