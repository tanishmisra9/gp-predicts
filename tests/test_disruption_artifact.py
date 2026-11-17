"""E205 independent raw-window counts and scalar earlier-only team features."""
import json
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

import f1.ff1  # initialize the same cache root as the extractor

from experiments.e190_traffic_feasibility import cached_session
from experiments.e205_disruption_winner import NEW,add_history
from experiments.e168_winner import win_score
from f1.data import DATA,load_results
from f1.features import build_features,pre_weekend_cols


def test_source():
    source=pd.read_csv('results/e205_disruption_source.csv')
    report=json.loads(Path('results/e205_source_summary.json').read_text())
    fastf1.Cache.offline_mode(True)
    for event in report['events']:
        if not event['available']:
            assert (event['season'],event['round'])==(2018,14)
            continue
        raw=pd.DataFrame(cached_session(fastf1,event['season'],event['round']).laps)
        actual=[]
        for driver,g in raw.dropna(subset=['Driver']).groupby('Driver',sort=True):
            g=g.sort_values('LapNumber'); seconds=g.LapTime.dt.total_seconds()
            good=g.LapNumber.gt(1)&g.PitInTime.isna()&g.PitOutTime.isna()&g.TrackStatus.astype(str).eq('1')&g.IsAccurate.fillna(False)&np.isfinite(seconds)&seconds.gt(0)
            valid=good&good.shift(1,fill_value=False)&good.shift(-1,fill_value=False)
            valid&=g.LapNumber.sub(g.LapNumber.shift()).eq(1)&g.LapNumber.shift(-1).sub(g.LapNumber).eq(1)
            valid&=g.Stint.notna()&g.Stint.eq(g.Stint.shift())&g.Stint.eq(g.Stint.shift(-1))
            valid&=g.Compound.isin(['SOFT','MEDIUM','HARD'])&g.Compound.eq(g.Compound.shift())&g.Compound.eq(g.Compound.shift(-1))
            age=g.TyreLife.to_numpy(float)
            age_ok=np.zeros(len(g),bool)
            if len(g)>2:
                age_ok[1:-1]=np.isclose(age[1:-1]-age[:-2],1)&np.isclose(age[2:]-age[1:-1],1)
            valid&=age_ok
            slow=seconds.ge(1.02*(seconds.shift()+seconds.shift(-1))/2)
            actual.append(dict(code=driver,eligible=int(valid.sum()),disruptions=int((valid&slow).sum())))
        expected=pd.DataFrame(actual).sort_values('code').reset_index(drop=True)
        saved=source[source.season.eq(event['season'])&source['round'].eq(event['round'])][expected.columns].sort_values('code').reset_index(drop=True)
        pd.testing.assert_frame_equal(saved,expected)
        assert event['drivers']==len(expected) and event['eligible']==expected.eligible.sum() and event['disruptions']==expected.disruptions.sum()
    assert sum(e['available'] for e in report['events'])>=95 and source.eligible.ge(3).mean()>=.8 and report['coverage_gate']


def test_features():
    df=load_results().loc[lambda d:d.season.le(2022)]
    pace=pd.read_csv(DATA/'racepace_ff1.csv').loc[lambda d:d.season.le(2022)]
    X=build_features(df,pace); base=pre_weekend_cols(X)
    source=pd.read_csv('results/e205_disruption_source.csv')
    observed=add_history(X,source)
    saved=pd.read_csv('results/e205_features.csv',float_precision='round_trip')
    np.testing.assert_allclose(observed[NEW],saved[NEW],atol=1e-14)
    meta={(r.season,r.round,r.code):(r.k,r.team) for r in X.itertuples()}
    records=[(meta[(r.season,r.round,r.code)],r.eligible,r.disruptions) for r in source.itertuples()]
    for r in X.itertuples():
        numerator=.5; denominator=1.; team_n=0.; team_d=0.
        for (k,team),eligible,disruptions in records:
            if k<r.k:
                w=2**(-(r.k-k)/10)
                numerator+=w*disruptions; denominator+=w*eligible
                if team==r.team:
                    team_n+=w*disruptions; team_d+=w*eligible
        expected=(team_n+100*numerator/denominator)/(team_d+100)
        np.testing.assert_allclose(observed.loc[r.Index,NEW[0]],expected,atol=1e-14)
    target=X[X.season.eq(2022)&X['round'].eq(10)]; cutoff=target.k.iloc[0]
    changed=source.copy()
    future=[meta[(r.season,r.round,r.code)][0]>=cutoff for r in changed.itertuples()]
    changed.loc[future,'disruptions']=changed.loc[future,'eligible']
    again=add_history(X,changed)
    np.testing.assert_array_equal(observed.loc[X.k.le(cutoff),NEW],again.loc[X.k.le(cutoff),NEW])
    train=observed[observed.k.lt(cutoff)&observed.season.ge(2019)]
    other_train=again[again.k.lt(cutoff)&again.season.ge(2019)]
    current=observed.loc[target.index]
    other_current=again.loc[target.index].assign(gridn=-999,order=99,win=0)
    np.testing.assert_array_equal(win_score(train,current,base+NEW),win_score(other_train,other_current,base+NEW))
    training=X.k.lt(cutoff)&X.season.ge(2019); current=X.k.eq(cutoff)
    F=base+NEW
    original_score=win_score(observed[training],observed[current],F)
    corrupted=again[current].assign(order=99,win=0,gridn=-999,podium=0)
    np.testing.assert_array_equal(original_score,win_score(again[training],corrupted,F))
    F=base+NEW; train=observed[observed.k.lt(cutoff)&observed.season.ge(2019)]
    a=win_score(train,observed.loc[target.index],F)
    b=win_score(again.loc[train.index],again.loc[target.index].assign(gridn=-999),F)
    np.testing.assert_array_equal(a,b)
    earlier=source.copy(); row=earlier[earlier.season.eq(2021)&earlier.eligible.gt(0)].index[0]
    earlier.loc[row,'disruptions']=earlier.loc[row,'eligible']
    assert not add_history(X,earlier).loc[target.index,NEW].equals(observed.loc[target.index,NEW])


if __name__=='__main__':
    test_source()
    test_features()
    print('E205 independent raw windows/scalar histories, exact target/future score invariance and earlier responsiveness PASS')
