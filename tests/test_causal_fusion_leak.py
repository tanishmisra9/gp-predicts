"""E188 scales and forecast states must not depend on target/future data."""
import numpy as np
import pandas as pd
from experiments.e188_causal_fusion import INDICATORS,centered_team_table,filter_teams,freeze_scales,source_table


def test_calibration():
    t=pd.DataFrame(dict(season=[2018,2018,2019,2019,2020,2020],round=[1]*6,team=['a','b']*3))
    for c in INDICATORS:
        t[c]=[-1.,1.,-2.,2.,-3.,3.]
    scale=freeze_scales(t)
    np.testing.assert_allclose(scale,np.sqrt(2.))
    old=t[INDICATORS].std()
    bad=t.copy(); bad.loc[bad.season.ge(2020),INDICATORS]*=100
    assert not old.equals(bad[INDICATORS].std())
    pd.testing.assert_series_equal(freeze_scales(t),freeze_scales(bad),check_exact=True)
    pd.testing.assert_series_equal(freeze_scales(t,['pace_gap']),scale[['pace_gap']],check_exact=True)
    before=filter_teams(t,INDICATORS,2.)
    after=filter_teams(bad,INDICATORS,2.)
    pd.testing.assert_frame_equal(before,after,check_exact=True)
    assert before.season.min()==2019
    bad.loc[bad.season.eq(2019),INDICATORS]*=-10
    after=filter_teams(bad,INDICATORS,2.)
    assert not before[before.season.eq(2020)].strength.equals(after[after.season.eq(2020)].strength)
    zero=t.copy(); zero.loc[zero.season.eq(2018),INDICATORS]=0.
    try:
        freeze_scales(zero)
    except AssertionError:
        pass
    else:
        raise AssertionError('zero calibration accepted')
    # One observation: m=0,P=3.03,r=1 then prediction from that posterior.
    single=filter_teams(t,['pace_gap'],1.)
    expected=3.03/(3.03+1)*(-1/np.sqrt(2))
    assert np.isclose(single[(single.season==2019)&single.team.eq('a')].strength.iloc[0],expected)


def test_real_source():
    t,_=source_table()
    before=filter_teams(t,INDICATORS,2.)
    bad=t.copy()
    mask=bad.season.gt(2021)|(bad.season.eq(2021)&bad['round'].ge(5))
    bad.loc[mask,INDICATORS]=999.
    after=filter_teams(bad,INDICATORS,2.)
    target=before.season.eq(2021)&before['round'].eq(5)
    pd.testing.assert_frame_equal(before.loc[target],after.loc[target],check_exact=True)
    bad.loc[bad.season.eq(2021)&bad['round'].eq(4),INDICATORS]=-999.
    after=filter_teams(bad,INDICATORS,2.)
    assert not before.loc[target,'strength'].equals(after.loc[target,'strength'])
    null=filter_teams(t,INDICATORS,2.,shuffled=True)
    pd.testing.assert_frame_equal(null,filter_teams(t,INDICATORS,2.,shuffled=True),check_exact=True)
    assert t.season.max()==2022


def test_raw_observations():
    r=pd.DataFrame(dict(season=[2018,2018,2019,2019,2020,2020],round=[1]*6,team=['a','b']*3,code=['A','B']*3,q1=[80.,82.]*3,points=[25.,18.]*3,order=[1,2]*3))
    keys=['season','round','code']
    pace=r[keys].assign(pace_gap=[0.,2.]*3)
    fuel=r[keys].assign(fcorr_gap=[0.,2.]*3)
    sessions=r[keys].assign(session='FP2',lr_gap=[0.,2.]*3)
    t=centered_team_table(r,pace,fuel,sessions)
    before=filter_teams(t,INDICATORS,2.)
    r.loc[r.season.eq(2020),['q1','points','order']]=999.
    pace.loc[pace.season.eq(2020),'pace_gap']=999.
    fuel.loc[fuel.season.eq(2020),'fcorr_gap']=999.
    sessions.loc[sessions.season.eq(2020),'lr_gap']=999.
    got=filter_teams(centered_team_table(r,pace,fuel,sessions),INDICATORS,2.)
    pd.testing.assert_frame_equal(before,got,check_exact=True)


if __name__=='__main__':
    test_calibration()
    test_real_source()
    test_raw_observations()
    print('global scaling leak reproduced; warmup calibration, update math and source cutoff passed')
