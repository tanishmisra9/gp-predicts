"""E235 independent entrant source mapping, scalar features and fitted timing."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from experiments.e168_winner import win_score
from experiments.e235_launch_winner import NEW,observations,add_history
from f1.data import DATA,load_results
from f1.features import build_features,pre_weekend_cols


def test_features():
    df=load_results().loc[lambda d:d.season.le(2022)]
    pace=pd.read_csv(DATA/'racepace_ff1.csv').loc[lambda d:d.season.le(2022)]
    X=build_features(df,pace); base=pre_weekend_cols(X)
    entries=df.loc[df.season.between(2018,2022),['season','round','code']]
    reports=json.loads(Path('results/e235_source_status.json').read_text())
    table=pd.read_csv('results/e235_launch_observations.csv',float_precision='round_trip')
    mapping={}
    for report in reports:
        for row in report.get('rows',[]):
            mapping[(report['season'],report['round'],row['driver'])]=row.get('relative20',np.nan) if report['gate'] and row['supported'] else np.nan
    expected=entries.copy().reset_index(drop=True)
    expected['response']=[mapping.get(tuple(row),np.nan) for row in expected.to_numpy()]
    pd.testing.assert_frame_equal(table,expected,atol=1e-14,rtol=1e-14)
    summary=json.loads(Path('results/e235_source_summary.json').read_text())
    available=sum(r['available'] for r in reports); supported=sum(r['gate'] for r in reports)
    coverage=float(expected.response.notna().mean())
    assert summary==dict(sessions=103,available=available,supported_events=supported,driver_coverage=coverage,gate=bool(available>=95 and supported>=80 and coverage>=.8))
    old=pd.read_csv('results/e234_driver_measurements.csv',float_precision='round_trip').rename(columns={'driver':'code'})
    now=table[table.season.eq(2019)]
    pd.testing.assert_frame_equal(now.sort_values(['round','code']).reset_index(drop=True),old.sort_values(['round','code']).reset_index(drop=True),atol=1e-14,rtol=1e-14)
    features=add_history(X,table)
    saved=pd.read_csv('results/e235_features.csv',float_precision='round_trip')
    np.testing.assert_array_equal(features[['season','round','code','k']],saved[['season','round','code','k']])
    np.testing.assert_allclose(features[NEW],saved[NEW],atol=1e-14,rtol=1e-14)
    event_k={(r.season,r.round):r.k for r in X[['season','round','k']].drop_duplicates().itertuples()}
    records=[(event_k[(r.season,r.round)],r.code,r.response) for r in table.itertuples()]
    for row in X.itertuples():
        valid=[(k,value) for k,code,value in records if code==row.code and k<row.k and np.isfinite(value)]
        numerator=sum(2**(-(row.k-k)/10)*value for k,value in valid)
        denominator=5+sum(2**(-(row.k-k)/10) for k,value in valid)
        np.testing.assert_allclose(features.loc[row.Index,NEW[0]],numerator/denominator,atol=1e-14,rtol=1e-14)
    reslice=add_history(X,table[table.season.eq(2019)])
    original=pd.read_csv('results/e234_temporal_forecasts.csv',float_precision='round_trip').rename(columns={'driver':'code'})
    joined=reslice[reslice.season.eq(2019)].merge(original[['season','round','code','prediction']],on=['season','round','code'],validate='one_to_one')
    np.testing.assert_allclose(joined[NEW[0]],joined.prediction,atol=1e-14,rtol=1e-14)
    target=X[X.season.eq(2022)&X['round'].eq(10)]; cutoff=target.k.iloc[0]
    changed_reports=json.loads(json.dumps(reports))
    for report in changed_reports:
        if event_k[(report['season'],report['round'])]>=cutoff:
            for row in report.get('rows',[]):
                if row['supported']: row['relative20']=999
    changed=add_history(X,observations(changed_reports,entries))
    np.testing.assert_array_equal(features.loc[X.k.le(cutoff),NEW],changed.loc[X.k.le(cutoff),NEW])
    prior_reports=[r for r in reports if event_k[(r['season'],r['round'])]<cutoff]
    prefix=add_history(X,observations(prior_reports,entries))
    np.testing.assert_array_equal(features.loc[target.index,NEW],prefix.loc[target.index,NEW])
    train=X.k.lt(cutoff)&X.season.ge(2019)
    a=win_score(features[train],features.loc[target.index],base+NEW)
    b=win_score(changed[train],changed.loc[target.index].assign(gridn=-999,order=99,win=0),base+NEW)
    np.testing.assert_array_equal(a,b)
    prediction=pd.read_csv('results/e235_inner_predictions.csv',float_precision='round_trip')
    prediction=prediction[prediction.season.eq(2022)&prediction['round'].eq(10)&prediction.model.eq('launch_win')].set_index('driver')
    np.testing.assert_allclose(a,prediction.loc[target.driver,'score'],atol=1e-14,rtol=1e-14)
    earlier=table.copy(); first=earlier[earlier.season.eq(2021)&earlier.response.notna()&earlier.code.isin(target.code)].index[-1]
    earlier.loc[first,'response']+=1
    assert not add_history(X,earlier).loc[target.index,NEW].equals(features.loc[target.index,NEW])


if __name__=='__main__':
    test_features()
    print('E235 independent source/coverage/scalar features/pilot parity/fitted timing PASS')
