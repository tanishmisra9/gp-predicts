"""E193 exact original reliability-source fidelity and E192 default preservation."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.e192_simulated_winner import DNF_F,add_reliability,simulate
from experiments.e193_complete_simulation import FULL_DNF_F,add_complete_reliability,supplier
from f1.features import build_features,pre_weekend_cols
from f1.profile import add_profile


def test_fidelity_and_default():
    module=ast.parse(Path('experiments/d_montecarlo.py').read_text())
    assignment=next(n for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DNF_F' for t in n.targets))
    original=[c.value for c in assignment.value.generators[0].iter.elts]
    assert FULL_DNF_F==original
    df=pd.read_csv('data/results.csv').loc[lambda d:d.season.le(2022)].copy()
    pace=pd.read_csv('data/racepace_ff1.csv').loc[lambda d:d.season.le(2022)]
    raw=build_features(df,pace); F=pre_weekend_cols(raw)
    complete=add_complete_reliability(raw,df)
    # Only circuit-layout placeholders satisfy unrelated profile output columns;
    # none can contribute to these four status-derived reliability quantities.
    circ=df[['season','round']].drop_duplicates().assign(c_slow=1.,c_med=1.,c_fast=1.,c_ft=.5)
    empty=pd.DataFrame()
    reference=add_profile(raw,df,aux=(empty,circ,empty,empty,empty,pace))
    cols=['d_coll','c_dnf','c_coll','pu_pufail']
    pd.testing.assert_frame_equal(complete[cols],reference[cols],check_exact=True)
    assert supplier(2019,'racing_point') is None and supplier(2020,'alfa') is None
    assert supplier(2020,'mercedes')=='mer'
    target=raw.season.eq(2021)&raw['round'].eq(5)
    bad=df.copy(); future=bad.season.gt(2021)|(bad.season.eq(2021)&bad['round'].ge(5))
    bad.loc[future,'status']='Engine'
    revised=add_complete_reliability(raw,bad)
    pd.testing.assert_frame_equal(complete.loc[target,cols],revised.loc[target,cols],check_exact=True)
    bad.loc[bad.season.eq(2020),'status']='Engine'
    earlier=add_complete_reliability(raw,bad)
    assert not complete.loc[target,'pu_pufail'].equals(earlier.loc[target,'pu_pufail'])
    reduced=add_reliability(raw,df).merge(pace[['season','round','code','pace_gap']].rename(columns={'pace_gap':'y_pace'}),on=['season','round','code'],how='left',validate='one_to_one')
    current=reduced[reduced.season.eq(2020)&reduced['round'].eq(5)]
    training=reduced[reduced.k.lt(current.k.iloc[0])&reduced.season.ge(2019)]
    default=simulate(training,current,F)
    explicit=simulate(training,current,F,reliability_features=DNF_F)
    pd.testing.assert_frame_equal(default,explicit,check_exact=True)
    saved=pd.read_csv('results/e192_inner_simulation_probabilities.csv',float_precision='round_trip')
    saved=saved[saved.season.eq(2020)&saved['round'].eq(5)&~saved['null']].drop(columns='null').reset_index(drop=True)
    pd.testing.assert_frame_equal(default.reset_index(drop=True),saved,check_exact=True)
    full=complete.merge(pace[['season','round','code','pace_gap']].rename(columns={'pace_gap':'y_pace'}),on=['season','round','code'],how='left',validate='one_to_one')
    current=full[full.season.eq(2020)&full['round'].eq(5)]
    training=full[full.k.lt(current.k.iloc[0])&full.season.ge(2019)]
    p=simulate(training,current,F,reliability_features=FULL_DNF_F)
    bad=current.copy(); bad[['order','win','podium','gridn','q_gap','y_pace','dnf_y']]=999.
    changed=simulate(training,bad,F,reliability_features=FULL_DNF_F)
    pd.testing.assert_frame_equal(p,changed,check_exact=True)


if __name__=='__main__':
    test_fidelity_and_default()
    print('original reliability columns exact; seven-feature default, source cutoff and full target invariance pass')
