"""Future completed calendar cannot determine past championship context."""
import numpy as np
import pandas as pd

from experiments.e187_context_audit import CONTEXT_FEATURES, add_context
from f1.features import build_features
import f1.walkforward as W


def test_future_calendar():
    x=pd.DataFrame(dict(season=[2020]*4,round=[1,1,2,2],k=[0,0,1,1],
        d_season_pts=[np.nan,np.nan,1.,.8],t_season_pts=[np.nan,np.nan,1.,1.]))
    future=x.iloc[-2:].assign(round=30,k=2)
    augmented=pd.concat([x,future],ignore_index=True)
    old=x.groupby('season')['round'].transform('max')-x['round']
    old_future=augmented.groupby('season')['round'].transform('max')-augmented['round']
    assert not old.equals(old_future.iloc[:len(x)])
    safe=add_context(x); changed=add_context(augmented)
    pd.testing.assert_frame_equal(safe,changed.iloc[:len(x)],check_exact=True)
    assert not {'nrounds','rounds_left','lateseason','title_live'}.intersection(safe.columns)
    assert safe.loc[safe.k.eq(0),CONTEXT_FEATURES].isna().all().all()
    np.testing.assert_allclose(safe.loc[safe.k.eq(1),'lead_margin'],.2)
    tied=add_context(x.assign(d_season_pts=1.))
    assert tied.lead_margin.eq(0).all()
    W.assert_pre_weekend(CONTEXT_FEATURES)


def test_real_context():
    df=pd.read_csv('data/results.csv').loc[lambda d:d.season.le(2022)].copy()
    pace=pd.read_csv('data/racepace_ff1.csv').loc[lambda d:d.season.le(2022)].copy()
    x=build_features(df,pace)
    baseline=add_context(x)
    mask=x.season.eq(2021)&x['round'].eq(5)
    k=x.loc[mask,'k'].iloc[0]
    future=x.k.gt(k)
    changed_x=x.copy()
    changed_x.loc[future,'round']=99
    changed=add_context(changed_x)
    pd.testing.assert_frame_equal(baseline.loc[mask,CONTEXT_FEATURES],changed.loc[mask,CONTEXT_FEATURES],check_exact=True)
    bad=df.copy(); altered=pace.copy()
    later=bad.season.gt(2021)|(bad.season.eq(2021)&bad['round'].ge(5))
    bad.loc[later,['points','grid','q_pos','q1','q2','q3']]=999.
    bad.loc[later,'order']=99
    later_p=altered.season.gt(2021)|(altered.season.eq(2021)&altered['round'].ge(5))
    altered.loc[later_p,'pace_gap']=999.
    got=add_context(build_features(bad,altered))
    pd.testing.assert_frame_equal(baseline.loc[mask,CONTEXT_FEATURES],got.loc[mask,CONTEXT_FEATURES],check_exact=True)
    earlier=bad.season.eq(2021)&bad['round'].eq(4)
    bad.loc[earlier,'points']=np.arange(earlier.sum(),dtype=float)
    got=add_context(build_features(bad,altered))
    assert not baseline.loc[mask,CONTEXT_FEATURES].equals(got.loc[mask,CONTEXT_FEATURES])


if __name__=='__main__':
    test_future_calendar()
    test_real_context()
    print('old calendar leak reproduced; safe context calendar/source mutation and missing/tie checks passed')
