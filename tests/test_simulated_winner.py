"""E192 simulator counts and strict prior-only source/model boundaries."""
import numpy as np
import pandas as pd

from experiments.e192_simulated_winner import add_reliability,probabilities_from_performance,simulate
from f1.features import build_features,pre_weekend_cols


def test_counts():
    performance=np.array([[0,1,2,3],[1,0,2,3],[np.inf,1,0,2],[np.inf]*4],float)
    win,podium,smooth,count=probabilities_from_performance(performance)
    assert count==3
    np.testing.assert_allclose(win,[1/3,1/3,1/3,0])
    np.testing.assert_allclose(podium,[2/3,1,1,1/3])
    np.testing.assert_allclose(smooth,[.3,.3,.3,.1])
    tied=probabilities_from_performance(np.zeros((1,3)))
    np.testing.assert_array_equal(tied[0],[1,0,0])
    try:
        probabilities_from_performance(np.full((3,2),np.inf))
    except AssertionError:
        pass
    else:
        raise AssertionError('all-retired simulation accepted')
    p=np.array([[1,0,3,2]]*3+[[1,3,0,2]]*2+[[0,3,2,1]],float)
    win,pod,_,_=probabilities_from_performance(p)
    assert win.argmax()==1 and pod.argmax()==0


def test_source_and_target():
    df=pd.read_csv('data/results.csv').loc[lambda d:d.season.le(2022)].copy()
    pace=pd.read_csv('data/racepace_ff1.csv').loc[lambda d:d.season.le(2022)]
    raw=build_features(df,pace); F=pre_weekend_cols(raw)
    X=add_reliability(raw,df)
    mask=X.season.eq(2021)&X['round'].eq(5)
    bad=df.copy(); future=bad.season.gt(2021)|(bad.season.eq(2021)&bad['round'].ge(5))
    bad.loc[future,'status']='Collision'
    got=add_reliability(raw,bad)
    pd.testing.assert_frame_equal(X.loc[mask,['d_coll','c_dnf']],got.loc[mask,['d_coll','c_dnf']],check_exact=True)
    bad.loc[bad.season.eq(2019),'status']='Collision'
    earlier=add_reliability(raw,bad)
    assert not X.loc[mask,['d_coll','c_dnf']].equals(earlier.loc[mask,['d_coll','c_dnf']])
    X=X.merge(pace[['season','round','code','pace_gap']].rename(columns={'pace_gap':'y_pace'}),on=['season','round','code'],how='left',validate='one_to_one')
    current=X[X.season.eq(2020)&X['round'].eq(5)]
    train=X[X.k.lt(current.k.iloc[0])&X.season.ge(2019)]
    before=simulate(train,current,F)
    changed=current.copy(); changed[['order','win','podium','q_gap','gridn','y_pace','dnf_y']]=999.
    after=simulate(train,changed,F)
    pd.testing.assert_frame_equal(before,after,check_exact=True)
    reordered=simulate(train.sample(frac=1,random_state=2),current.sample(frac=1,random_state=3),F)
    pd.testing.assert_frame_equal(before,reordered,check_exact=True)
    assert np.isclose(before.win_probability.sum(),1.)
    assert np.isclose(before.podium_probability.sum(),3.)
    try:
        simulate(train,current,F+['gridn'])
    except ValueError:
        pass
    else:
        raise AssertionError('forbidden target feature accepted')


if __name__=='__main__':
    test_counts(); test_source_and_target()
    print('simulator count distinction, ties, reliability-source cutoff and target-label/grid invariance passed')
