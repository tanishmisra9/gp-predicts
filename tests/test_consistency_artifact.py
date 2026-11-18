"""E214 independent raw selection, scalar OLS/MAD, histories and fitted timing."""
import json
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

import f1.ff1
from experiments.e168_winner import win_score
from experiments.e190_traffic_feasibility import cached_session
from experiments.e214_consistency_winner import NEW,add_history,event_measurements,stint_spreads
from f1.data import DATA,load_results
from f1.features import build_features,pre_weekend_cols


def test_artifacts():
    stints=pd.read_csv('results/e214_stint_spreads.csv',float_precision='round_trip')
    source=pd.read_csv('results/e214_consistency_source.csv',float_precision='round_trip')
    summary=json.loads(Path('results/e214_source_summary.json').read_text())
    fastf1.Cache.offline_mode(True); rows=[]
    for receipt in summary['events']:
        season=receipt['season']; round_=receipt['round']
        if not receipt['available']:
            assert stints[stints.season.eq(season)&stints['round'].eq(round_)].empty
            continue
        raw=pd.DataFrame(cached_session(fastf1,season,round_).laps)
        assert not raw.duplicated(['Driver','LapNumber']).any()
        valid=raw.LapNumber.gt(1)&raw.LapTime.notna()&raw.Time.notna()&raw.LapStartTime.notna()
        valid&=raw.PitInTime.isna()&raw.PitOutTime.isna()&raw.TrackStatus.astype(str).eq('1')&raw.IsAccurate.fillna(False)
        selected=raw[valid].copy(); selected['seconds']=selected.LapTime.dt.total_seconds()
        medians={d:np.median(g.seconds) for d,g in selected.groupby('Driver')}
        selected=selected[[r.seconds<1.07*medians[r.Driver] for r in selected.itertuples()]]
        count=0
        for (driver,stint,compound),g in selected.groupby(['Driver','Stint','Compound'],sort=True):
            if compound not in ('SOFT','MEDIUM','HARD') or len(g)<8:
                continue
            x=g.LapNumber.to_numpy(float); y=g.seconds.to_numpy(float)
            xm=sum(x)/len(x); ym=sum(y)/len(y)
            slope=sum((a-xm)*(b-ym) for a,b in zip(x,y))/sum((a-xm)**2 for a in x)
            errors=np.array([b-(ym+slope*(a-xm)) for a,b in zip(x,y)])
            middle=np.median(errors)
            spread=100*1.4826*np.median([abs(e-middle) for e in errors])/np.median(y)
            rows.append(dict(code=driver,stint=stint,compound=compound,laps=len(g),spread=spread,season=season,round=round_))
            count+=1
        assert count==receipt['stints']
    expected=pd.DataFrame(rows)
    keys=['season','round','code','stint','compound']
    pd.testing.assert_frame_equal(stints.sort_values(keys).reset_index(drop=True),expected[stints.columns].sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-10,rtol=1e-10)
    entries=load_results().loc[lambda d:d.season.between(2018,2022),['season','round','code']]
    values={(s,r,d):float(np.median(g.spread)) for (s,r,d),g in expected.groupby(['season','round','code'])}
    measured=entries.copy(); measured['spread']=[values.get((r.season,r.round,r.code),np.nan) for r in measured.itertuples()]
    measured['centered_spread']=np.nan
    for _,ids in measured.groupby(['season','round']).groups.items():
        v=measured.loc[ids,'spread']; good=v.notna()
        if good.sum()>=10:
            measured.loc[ids,'centered_spread']=v-v[good].mean()
    pd.testing.assert_frame_equal(source,measured,check_dtype=False,atol=1e-10,rtol=1e-10)
    assert len(source)==2060
    coverage=float(measured.centered_spread.notna().mean())
    np.testing.assert_allclose(summary['coverage'],coverage,atol=1e-15)
    assert summary['gate']==bool(sum(r['available'] for r in summary['events'])>=95 and coverage>=.8)
    df=load_results().loc[lambda d:d.season.le(2022)]
    pace=pd.read_csv(DATA/'racepace_ff1.csv').loc[lambda d:d.season.le(2022)]
    X=build_features(df,pace); base=pre_weekend_cols(X)
    featured=add_history(X,source)
    saved=pd.read_csv('results/e214_features.csv',float_precision='round_trip')
    np.testing.assert_array_equal(featured[['season','round','code','k']],saved[['season','round','code','k']])
    np.testing.assert_allclose(featured[NEW],saved[NEW],atol=1e-14)
    metadata={(r.season,r.round):r.k for r in X[['season','round','k']].drop_duplicates().itertuples()}
    obs=[(metadata[(r.season,r.round)],r.code,r.centered_spread) for r in measured.itertuples()]
    for r in X.itertuples():
        numerator=0.; denominator=5.
        for k,code,v in obs:
            if k<r.k and code==r.code and np.isfinite(v):
                w=2**(-(r.k-k)/10); numerator+=w*v; denominator+=w
        np.testing.assert_allclose(featured.loc[r.Index,NEW[0]],numerator/denominator,atol=1e-10)
    target=X[X.season.eq(2022)&X['round'].eq(10)]; cutoff=target.k.iloc[0]
    altered=stints.copy()
    later=[metadata[(r.season,r.round)]>=cutoff for r in stints.itertuples()]
    altered.loc[later,'spread']=99
    changed=add_history(X,event_measurements(altered,entries))
    np.testing.assert_array_equal(featured.loc[X.k.le(cutoff),NEW],changed.loc[X.k.le(cutoff),NEW])
    prefix=add_history(X,event_measurements(stints[np.logical_not(later)],entries))
    np.testing.assert_array_equal(featured.loc[target.index,NEW],prefix.loc[target.index,NEW])
    raw=pd.DataFrame(cached_session(fastf1,2022,10).laps).copy()
    raw['LapTime']+=pd.to_timedelta(np.sin(raw.LapNumber.to_numpy(float))*.5,unit='s')
    replacement=stint_spreads(raw).assign(season=2022,round=10)
    amended=pd.concat([stints[~(stints.season.eq(2022)&stints['round'].eq(10))],replacement],ignore_index=True)
    raw_changed=add_history(X,event_measurements(amended,entries))
    np.testing.assert_array_equal(featured.loc[X.k.le(cutoff),NEW],raw_changed.loc[X.k.le(cutoff),NEW])
    assert not raw_changed.loc[X.k.gt(cutoff),NEW].equals(featured.loc[X.k.gt(cutoff),NEW])
    training=X.k.lt(cutoff)&X.season.ge(2019)
    a=win_score(featured[training],featured.loc[target.index],base+NEW)
    b=win_score(changed[training],changed.loc[target.index].assign(gridn=-999,order=99,win=0),base+NEW)
    np.testing.assert_array_equal(a,b)
    pred=pd.read_csv('results/e214_inner_predictions.csv',float_precision='round_trip')
    pred=pred[pred.season.eq(2022)&pred['round'].eq(10)&pred.model.eq('consistency_win')].set_index('driver')
    np.testing.assert_allclose(a,pred.loc[target.driver,'score'],atol=1e-14,rtol=1e-14)
    altered=stints.copy(); old=stints[stints.season.eq(2021)].iloc[0]
    altered.loc[altered.season.eq(2021)&altered['round'].eq(old['round'])&altered.code.eq(old.code),'spread']+=1
    assert not add_history(X,event_measurements(altered,entries)).loc[target.index,NEW].equals(featured.loc[target.index,NEW])


if __name__=='__main__':
    test_artifacts()
    print('E214 independent all-source OLS/MAD/event aggregates, source gate, scalar histories and fitted timing PASS')
