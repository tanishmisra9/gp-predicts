"""Independent raw Q1 sector reductions and scalar historical features."""
import json
from pathlib import Path
from unittest.mock import patch

import fastf1
import numpy as np
import pandas as pd

import f1.ff1
from experiments.e168_winner import win_score
from experiments.e208_ideal_q1 import NEW,add_history
from f1.data import DATA,load_results
from f1.features import build_features,pre_weekend_cols


def test_source():
    source=pd.read_csv('results/e208_q1_source.csv',float_precision='round_trip')
    summary=json.loads(Path('results/e208_source_summary.json').read_text())
    fastf1.Cache.offline_mode(True); tables=[]
    for event in summary['sessions']:
        with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('audit HTTP forbidden')):
            session=fastf1.get_session(event['season'],event['round'],'Q')
            session.load(laps=True,telemetry=False,weather=False,messages=False)
            q1,_,_=session.laps.split_qualifying_sessions()
        assert event['available'] and q1 is not None
        # Independently check phase boundaries when parser split times exist.
        splits=session._session_split_times
        if splits and len(splits)>=2:
            assert (q1.LapStartTime>splits[0]).all() and (q1.LapStartTime<splits[1]).all()
        samples={}
        for r in q1.itertuples():
            times=[r.LapTime,r.Sector1Time,r.Sector2Time,r.Sector3Time]
            if any(pd.isna(t) for t in times):
                continue
            values=np.array([t.total_seconds() for t in times])
            if not np.isfinite(values).all() or not (values>0).all() or not np.isclose(sum(values[1:]),values[0],rtol=0,atol=.010):
                continue
            if pd.notna(r.PitInTime) or pd.notna(r.PitOutTime) or str(r.TrackStatus)!='1' or pd.isna(r.IsAccurate) or not r.IsAccurate:
                continue
            if r.Compound not in {'HYPERSOFT','ULTRASOFT','SUPERSOFT','SOFT','MEDIUM','HARD','SUPERHARD'}:
                continue
            deleted=getattr(r,'Deleted',False)
            if pd.notna(deleted) and bool(deleted):
                continue
            samples.setdefault(r.Driver,{}).setdefault(r.Compound,[]).append(values)
        rows=[]
        for driver,compounds in sorted(samples.items()):
            ideals=[]; actuals=[]; count=0
            for laps in compounds.values():
                array=np.array(laps); count+=len(array)
                ideals.append(sum(min(array[:,i]) for i in (1,2,3)))
                actuals.extend(array[:,0])
            rows.append(dict(code=driver,ideal_seconds=min(ideals),best_seconds=min(actuals),eligible_laps=count))
        expected=pd.DataFrame(rows,columns=['code','ideal_seconds','best_seconds','eligible_laps'])
        for value,column in [('ideal_seconds','ideal_relative'),('best_seconds','actual_relative')]:
            expected[column]=[(v-np.median(expected[value]))/min(expected[value])*100 for v in expected[value]] if len(expected)>=10 else np.nan
        expected=expected.assign(season=event['season'],round=event['round'])
        saved=source[source.season.eq(event['season'])&source['round'].eq(event['round'])]
        pd.testing.assert_frame_equal(saved.reset_index(drop=True),expected,check_dtype=False,atol=1e-12,rtol=1e-12)
        assert event['drivers']==len(expected) and event['supported']==(len(expected)>=10) and event['eligible_laps']==expected.eligible_laps.sum()
        tables.append(expected)
    all_rows=pd.concat(tables,ignore_index=True)
    assert summary['available']==len(tables)==103
    assert summary['supported']==sum(len(t)>=10 for t in tables)
    np.testing.assert_allclose(summary['entrant_coverage'],all_rows.ideal_relative.notna().sum()/(20*103))
    assert summary['coverage_gate']==bool(len(tables)>=95 and summary['supported']>=80 and summary['entrant_coverage']>=.7)


def test_features():
    df=load_results().loc[lambda d:d.season.le(2022)]
    pace=pd.read_csv(DATA/'racepace_ff1.csv').loc[lambda d:d.season.le(2022)]
    X=build_features(df,pace); base=pre_weekend_cols(X)
    source=pd.read_csv('results/e208_q1_source.csv',float_precision='round_trip')
    result=add_history(X,source)
    saved=pd.read_csv('results/e208_features.csv',float_precision='round_trip')
    np.testing.assert_array_equal(result[['season','round','code','k']],saved[['season','round','code','k']])
    np.testing.assert_allclose(result[NEW],saved[NEW],atol=1e-14)
    metadata={(r.season,r.round,r.code):(r.k,r.team) for r in X.itertuples()}
    from collections import defaultdict
    observations=defaultdict(list)
    for r in source.itertuples():
        k,team=metadata[(r.season,r.round,r.code)]
        observations[(k,team)].append([r.ideal_relative,r.actual_relative])
    records=[]
    for (k,team),values in observations.items():
        array=np.array(values)
        mean=[np.mean(array[np.isfinite(array[:,i]),i]) if np.isfinite(array[:,i]).any() else np.nan for i in range(2)]
        records.append((k,team,mean))
    for r in X.itertuples():
        for side,column in enumerate(NEW):
            numerator=0.; denominator=5.
            for k,team,mean in records:
                if team==r.team and k<r.k and np.isfinite(mean[side]):
                    weight=2**(-(r.k-k)/10); numerator+=weight*mean[side]; denominator+=weight
            np.testing.assert_allclose(result.loc[r.Index,column],numerator/denominator,atol=1e-14)
    current=X[X.season.eq(2022)&X['round'].eq(10)]; cutoff=current.k.iloc[0]
    changed=source.copy(); fields=['ideal_relative','actual_relative']
    future=[metadata[(r.season,r.round,r.code)][0]>=cutoff for r in changed.itertuples()]
    changed.loc[future,fields]=999.
    again=add_history(X,changed)
    np.testing.assert_array_equal(result.loc[X.k.le(cutoff),NEW],again.loc[X.k.le(cutoff),NEW])
    train=X.k.lt(cutoff)&X.season.ge(2019); target=X.k.eq(cutoff)
    for field in NEW:
        np.testing.assert_array_equal(win_score(result[train],result[target],base+[field]),win_score(again[train],again[target].assign(order=99,win=0,gridn=-999),base+[field]))
    prior=source.copy(); row=prior[prior.season.eq(2021)&prior.code.isin(current.code)&prior.ideal_relative.notna()].index[-1]
    prior.loc[row,fields]+=10
    assert not add_history(X,prior).loc[current.index,NEW].equals(result.loc[current.index,NEW])


if __name__=='__main__':
    test_source(); test_features()
    print('E208 independent Q1 source/sectors/normalization/scalar histories and fitted timing audit PASS')
