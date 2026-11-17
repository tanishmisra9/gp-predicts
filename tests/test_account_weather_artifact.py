"""E197 saved weather source, independently matched timestamps and coverage."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.e190_traffic_feasibility import PILOT,cached_session


def test_artifacts():
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True)
    summary=json.loads(Path('results/e197_weather_source_summary.json').read_text())
    samples=pd.read_csv('results/e197_weather_samples.csv',float_precision='round_trip')
    laps=pd.read_csv('results/e197_weather_aligned_laps.csv',float_precision='round_trip')
    samples['Time']=pd.to_timedelta(samples.Time)
    laps['LapStartTime']=pd.to_timedelta(laps.LapStartTime)
    laps['weather_time']=pd.to_timedelta(laps.weather_time)
    assert [(r['season'],r['round']) for r in summary['sessions']]==PILOT
    assert set(map(tuple,samples[['season','round']].drop_duplicates().to_numpy()))==set(PILOT)
    assert set(map(tuple,laps[['season','round']].drop_duplicates().to_numpy()))==set(PILOT)
    assert not samples.duplicated(['season','round','Time']).any()
    assert not laps.duplicated(['season','round','Driver','LapNumber']).any()
    assert summary['authenticated'] is True
    reports=[]
    for report in summary['sessions']:
        season,round_=report['season'],report['round']
        weather=samples[samples.season.eq(season)&samples['round'].eq(round_)].sort_values('Time').reset_index(drop=True)
        current=laps[laps.season.eq(season)&laps['round'].eq(round_)].copy()
        # Reconstruct the clean-lap set without the extractor's filter helper.
        raw=pd.DataFrame(cached_session(fastf1,season,round_).laps)
        mask=(raw.LapNumber>1)&raw.LapTime.notna()&raw.Time.notna()&raw.LapStartTime.notna()&raw.PitInTime.isna()&raw.PitOutTime.isna()&raw.TrackStatus.astype(str).eq('1')&raw.IsAccurate.fillna(False)
        clean=raw[mask].copy(); clean['seconds']=clean.LapTime.dt.total_seconds()
        clean=clean[clean.seconds<1.07*clean.groupby('Driver').seconds.transform('median')]
        left=clean.set_index(['Driver','LapNumber']).sort_index()
        right=current.set_index(['Driver','LapNumber']).sort_index()
        assert left.index.equals(right.index)
        np.testing.assert_allclose(left.seconds,right.seconds,atol=1e-12)
        assert left.LapStartTime.equals(right.LapStartTime)
        # Independent binary search for the latest measurement before each lap.
        sample_time=weather.Time.dt.total_seconds().to_numpy()
        starts=current.LapStartTime.dt.total_seconds().to_numpy()
        index=np.searchsorted(sample_time,starts,side='right')-1
        matched=(index>=0)&(starts-sample_time[np.maximum(index,0)]<=120)
        assert np.all(matched)==current.weather_time.notna().all()
        chosen=weather.iloc[index[matched]]
        np.testing.assert_allclose(current.loc[matched,'weather_time'].dt.total_seconds(),chosen.Time.dt.total_seconds(),atol=1e-12)
        np.testing.assert_allclose(current.loc[matched,'TrackTemp'],chosen.TrackTemp,atol=1e-12)
        np.testing.assert_array_equal(current.loc[matched,'Rainfall'],chosen.Rainfall)
        finite=np.isfinite(current.TrackTemp.to_numpy())
        matched=matched&finite
        dry=matched&current.Rainfall.eq(False).to_numpy()
        span=float(np.percentile(current.loc[dry,'TrackTemp'],95)-np.percentile(current.loc[dry,'TrackTemp'],5))
        counts=current[dry].groupby('Driver').size()
        expected=dict(weather_rows=len(weather),finite_track_fraction=float(np.isfinite(weather.TrackTemp).mean()),clean_laps=len(current),matched_laps=int(matched.sum()),match_fraction=float(matched.mean()),dry_laps=int(dry.sum()),drivers_with_20_dry=int(counts.ge(20).sum()),dry_track_p95_p05=span)
        for name,value in expected.items():
            assert np.isclose(value,report[name],atol=1e-12),(name,value,report[name])
        assert report['available']
        assert len(report['requests'])==1 and report['requests'][0]['status']==200
        assert report['requests'][0]['url'].startswith('https://livetiming.formula1.com/static/2019/')
        assert report['requests'][0]['url'].endswith('/WeatherData.jsonStream')
        reports.append(expected)
    gate=all(r['weather_rows']>=100 and r['finite_track_fraction']>=.95 and r['match_fraction']>=.9 for r in reports)
    gate=gate and sum(r['dry_track_p95_p05']>=3 and r['drivers_with_20_dry']>=10 for r in reports)>=2
    assert summary['gate']==gate
    assert len(samples)==373 and len(laps)==3151


if __name__=='__main__':
    test_artifacts()
    print('E197 raw clean-lap set, binary-search weather alignment, coverage/ranges and gate reproduce')
