"""Independent scalar restart histories and real source-to-model timing audit."""
from collections import defaultdict

import numpy as np
import pandas as pd

from experiments.e168_winner import win_score
from experiments.e207_restart_winner import NEW,add_history
from f1.data import DATA,load_results
from f1.features import build_features,pre_weekend_cols


def test_features():
    df=load_results().loc[lambda d:d.season.le(2022)]
    pace=pd.read_csv(DATA/'racepace_ff1.csv').loc[lambda d:d.season.le(2022)]
    X=build_features(df,pace); base=pre_weekend_cols(X)
    source=pd.read_csv('results/e207_restart_source.csv',float_precision='round_trip')
    observed=add_history(X,source)
    saved=pd.read_csv('results/e207_features.csv',float_precision='round_trip')
    np.testing.assert_array_equal(observed[['season','round','code','k']],saved[['season','round','code','k']])
    np.testing.assert_allclose(observed[NEW],saved[NEW],atol=1e-14)
    metadata={(r.season,r.round,r.code):r.k for r in X.itertuples()}
    event_values=defaultdict(list)
    for r in source.itertuples():
        event_values[(metadata[(r.season,r.round,r.driver)],r.driver)].append(r.gain)
    records=defaultdict(list)
    for (k,driver),values in event_values.items():
        records[driver].append((k,float(np.mean(values))))
    for r in X.itertuples():
        numerator=0.; denominator=5.
        for prior_k,gain in records[r.code]:
            if prior_k<r.k:
                weight=2**(-(r.k-prior_k)/10)
                numerator+=weight*gain; denominator+=weight
        np.testing.assert_allclose(observed.loc[r.Index,NEW[0]],numerator/denominator,atol=1e-14)
    current=X[X.season.eq(2022)&X['round'].eq(10)]; cutoff=current.k.iloc[0]
    altered=source.copy()
    later=[metadata[(r.season,r.round,r.driver)]>=cutoff for r in source.itertuples()]
    altered.loc[later,'gain']=999.
    again=add_history(X,altered)
    np.testing.assert_array_equal(observed.loc[X.k.le(cutoff),NEW],again.loc[X.k.le(cutoff),NEW])
    train=X.k.lt(cutoff)&X.season.ge(2019); target=X.k.eq(cutoff)
    a=win_score(observed[train],observed[target],base+NEW)
    b=win_score(again[train],again[target].assign(order=99,win=0,gridn=-999),base+NEW)
    np.testing.assert_array_equal(a,b)
    prefix=source[[metadata[(r.season,r.round,r.driver)]<cutoff for r in source.itertuples()]]
    np.testing.assert_array_equal(add_history(X,prefix).loc[current.index,NEW],observed.loc[current.index,NEW])
    earlier=source.copy()
    index=next(i for i,r in earlier.iterrows() if r.driver in set(current.code) and metadata[(r.season,r['round'],r.driver)]<cutoff)
    earlier.loc[index,'gain']+=10
    assert not add_history(X,earlier).loc[current.index,NEW].equals(observed.loc[current.index,NEW])


if __name__=='__main__':
    test_features()
    print('E207 independent event means/scalar history, source timing and fitted-score invariance PASS')
