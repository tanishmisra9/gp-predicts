"""E201 categories use only training vocabularies and no target-weekend labels."""
import numpy as np
import pandas as pd

from experiments.e201_categorical_winner import CATS,attach_context,category_inputs,categorical_score,labels
from f1.data import DATA,load_results
from f1.features import build_features,pre_weekend_cols


def test_categories():
    train=pd.DataFrame(dict(k=np.repeat(np.arange(6),20),driver=[f'd{i}' for i in range(20)]*6,team=[f't{i//2}' for i in range(20)]*6,circuit_id=np.repeat(['c0','c1','c2']*2,20),d10_order=list(range(1,21))*6,win=([1]+[0]*19)*6))
    current=pd.DataFrame(dict(k=[6,6],driver=['d0','新-driver'],team=['t0','new-team'],circuit_id=['c0','new-circuit'],d10_order=[1.,20.],win=[0,1],gridn=[20,1]))
    a,b=category_inputs(train,current,['d10_order'])
    for name in CATS:
        assert list(a[name].cat.categories)==sorted(train[name].unique())
        assert list(b[name].cat.categories)==list(a[name].cat.categories)
        assert b[name].cat.codes.iloc[1]==-1
    y=labels(train,True)
    assert pd.Series(y).groupby(train.k).sum().eq(1).all()
    np.testing.assert_array_equal(labels(train.iloc[:80],True),y[:80])
    np.testing.assert_array_equal(labels(train,False),train.win)
    before=categorical_score(train,current,['d10_order'])
    after=categorical_score(train,current.assign(win=[1,0],gridn=[1,20]),['d10_order'])
    np.testing.assert_array_equal(before,after)
    assert np.isfinite(before).all() and (before<=0).all() and (before>=-1).all()
    earlier=train.copy(); earlier.loc[earlier.driver.eq('d0'),'driver']='renamed'
    _,changed=category_inputs(earlier,current,['d10_order'])
    assert changed.driver.cat.codes.iloc[0]==-1 and b.driver.cat.codes.iloc[0]>=0
    for tr,te,F,error in ((train.iloc[:0],current,['d10_order'],AssertionError),(train,current.assign(k=5),['d10_order'],AssertionError),(train,current,['gridn'],ValueError)):
        try:
            category_inputs(tr,te,F)
        except error:
            pass
        else:
            raise AssertionError('invalid temporal/feature input accepted')
    X=pd.DataFrame(dict(season=[2020],round=[1]))
    context=pd.DataFrame(dict(season=[2020],round=[1],circuit=['track']))
    assert attach_context(X,context).circuit_id.iloc[0]=='track'
    try:
        attach_context(X,pd.concat([context,context.assign(circuit='other')]))
    except AssertionError:
        pass
    else:
        raise AssertionError('conflicting event circuit accepted')


def test_real_source_cutoff():
    results=load_results().loc[lambda d:d.season.le(2022)]
    pace=pd.read_csv(DATA/'racepace_ff1.csv').loc[lambda d:d.season.le(2022)]
    before=attach_context(build_features(results,pace),results); F=pre_weekend_cols(before)
    target=before[before.season.eq(2022)&before['round'].eq(10)]
    k=target.k.iloc[0]; train=before[before.k.lt(k)&before.season.ge(2019)]
    changed=results.copy(); mask=changed.season.gt(2022)|(changed.season.eq(2022)&changed['round'].ge(10))
    for col in ('order','finish','grid','q_pos','q1','q2','q3','s_grid','s_order','s_points','points','laps'):
        changed.loc[mask,col]=999
    changed.loc[mask,'status']='Retired'
    altered=pace.copy(); future=altered.season.gt(2022)|(altered.season.eq(2022)&altered['round'].ge(10))
    altered.loc[future,'pace_gap']=np.random.default_rng(201).uniform(0,20,future.sum())
    after=attach_context(build_features(changed,altered),changed)
    now=after[after.k.eq(k)]; prior=after[after.k.lt(k)&after.season.ge(2019)]
    pd.testing.assert_frame_equal(target[F+CATS],now[F+CATS],check_exact=True)
    pd.testing.assert_frame_equal(train[F+CATS+['win']],prior[F+CATS+['win']],check_exact=True)
    np.testing.assert_array_equal(categorical_score(train,target,F),categorical_score(prior,now,F))
    earlier=results.copy(); old=earlier.season.eq(2022)&earlier['round'].eq(9)
    earlier.loc[old,'points']=100.
    previous=attach_context(build_features(earlier,pace),earlier)
    assert not target[F].equals(previous[previous.k.eq(k)][F])


if __name__=='__main__':
    test_categories(); test_real_source_cutoff()
    print('native category vocab/unseen/identity, null-prefix, guards and real current/future result+pace invariance passed')
