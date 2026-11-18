"""E212 independent team observations, scalar prior features and fitted timing."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.e168_winner import win_score
from experiments.e212_braking_winner import NEW,add_history,observations
from f1.data import DATA,load_results
from f1.features import build_features,pre_weekend_cols


def test_features():
    df=load_results().loc[lambda d:d.season.le(2022)]
    pace=pd.read_csv(DATA/'racepace_ff1.csv').loc[lambda d:d.season.le(2022)]
    X=build_features(df,pace); base=pre_weekend_cols(X)
    entries=df[['season','round','code','team']]
    laps=pd.read_csv('results/e212_braking_laps.csv',float_precision='round_trip')
    table=pd.read_csv('results/e212_braking_observations.csv',float_precision='round_trip')
    values=defaultdict(list)
    for r in laps.itertuples():
        values[(r.season,r.round,r.driver)].append(r.duration)
    rows=[]
    for (season,round_,team),event in entries.groupby(['season','round','team'],sort=True):
        driver_values=[float(np.mean(values[(season,round_,d)])) if len(values[(season,round_,d)])>=10 else np.nan for d in sorted(event.code.unique())]
        rows.append(dict(season=season,round=round_,team=team,response=float(np.mean(driver_values)) if len(driver_values)==2 and np.isfinite(driver_values).all() else np.nan))
    expected=pd.DataFrame(rows); expected['centered_response']=np.nan
    for _,ids in expected.groupby(['season','round']).groups.items():
        v=expected.loc[ids,'response']; valid=v.notna()
        if valid.sum()>=6:
            expected.loc[ids,'centered_response']=v-v[valid].mean()
    pd.testing.assert_frame_equal(table,expected,check_dtype=False,atol=1e-12,rtol=1e-12)
    receipts=json.loads(Path('results/e212_source_status.json').read_text())
    available=sum(r['available'] for r in receipts)
    support=int(expected.groupby(['season','round']).centered_response.count().ge(6).sum())
    coverage=float(expected.centered_response.notna().mean())
    summary=json.loads(Path('results/e212_source_summary.json').read_text())
    assert len(expected)==1030 and len(receipts)==103
    assert summary==dict(sessions=103,available=available,supported_events=support,team_coverage=coverage,
        gate=bool(available>=95 and support>=80 and coverage>=.8))
    previous=pd.read_csv('results/e211_team_measurements.csv',float_precision='round_trip')
    cols=['season','round','team','response','centered_response']
    pd.testing.assert_frame_equal(table[table.season.eq(2019)].reset_index(drop=True),previous[cols],check_dtype=False,atol=1e-12,rtol=1e-12)
    features=add_history(X,table)
    saved=pd.read_csv('results/e212_features.csv',float_precision='round_trip')
    np.testing.assert_array_equal(features[['season','round','code','k']],saved[['season','round','code','k']])
    np.testing.assert_allclose(features[NEW],saved[NEW],atol=1e-14)
    metadata={(r.season,r.round):r.k for r in X[['season','round','k']].drop_duplicates().itertuples()}
    records=[(metadata[(r.season,r.round)],r.team,r.centered_response) for r in table.itertuples()]
    for r in X.itertuples():
        numerator=0.; denominator=5.
        for k,team,value in records:
            if team==r.team and k<r.k and np.isfinite(value):
                w=2**(-(r.k-k)/10); numerator+=w*value; denominator+=w
        np.testing.assert_allclose(features.loc[r.Index,NEW[0]],numerator/denominator,atol=1e-14)
    original=pd.read_csv('results/e211_temporal_forecasts.csv',float_precision='round_trip')
    reslice=add_history(X,table[table.season.eq(2019)])
    check=reslice[reslice.season.eq(2019)].merge(original[['round','team','prediction']],on=['round','team'],validate='many_to_one')
    np.testing.assert_allclose(check[NEW[0]],check.prediction,atol=1e-14)
    target=X[X.season.eq(2022)&X['round'].eq(10)]; cutoff=target.k.iloc[0]
    later=[metadata[(r.season,r.round)]>=cutoff for r in laps.itertuples()]
    altered=laps.copy(); altered.loc[later,'duration']=2.8
    changed=add_history(X,observations(altered,entries))
    np.testing.assert_array_equal(features.loc[X.k.le(cutoff),NEW],changed.loc[X.k.le(cutoff),NEW])
    prefix=add_history(X,observations(laps[np.logical_not(later)],entries))
    np.testing.assert_array_equal(features.loc[target.index,NEW],prefix.loc[target.index,NEW])
    train=X.k.lt(cutoff)&X.season.ge(2019)
    a=win_score(features[train],features.loc[target.index],base+NEW)
    b=win_score(changed[train],changed.loc[target.index].assign(gridn=-999,order=99,win=0),base+NEW)
    np.testing.assert_array_equal(a,b)
    pred=pd.read_csv('results/e212_inner_predictions.csv',float_precision='round_trip')
    pred=pred[pred.season.eq(2022)&pred['round'].eq(10)&pred.model.eq('braking_win')].set_index('driver')
    np.testing.assert_allclose(a,pred.loc[target.driver,'score'],atol=1e-14,rtol=1e-14)
    early=laps.copy(); row=table[table.season.eq(2021)&table.centered_response.notna()].iloc[-1]
    codes=entries[entries.season.eq(2021)&entries['round'].eq(row['round'])&entries.team.eq(row.team)].code
    early.loc[early.season.eq(2021)&early['round'].eq(row['round'])&early.driver.isin(codes),'duration']+=.5
    assert not add_history(X,observations(early,entries)).loc[target.index,NEW].equals(features.loc[target.index,NEW])


if __name__=='__main__':
    test_features()
    print('E212 independent multi-year observations/scalar histories/pilot parity and exact fitted timing PASS')
