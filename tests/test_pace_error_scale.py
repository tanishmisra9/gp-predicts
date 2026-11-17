"""E194 error centering, shrinkage and strictly earlier variance-source use."""
import numpy as np
import pandas as pd

from experiments.e194_pace_error_scale import forecast_scales,pace_mean,revealed_errors
from f1.features import build_features,pre_weekend_cols


def test_math_and_cutoff():
    observations=pd.DataFrame(dict(k=[0,0,1,1,2,2],team=['a','b']*3,team_mse=[4.,1.,3.,2.,9.,8.]))
    entries=pd.DataFrame(dict(k=[0,0,1,1,1,2,2,3,3],team=['a','b','a','b','new','a','b','a','b']))
    out=forecast_scales(entries,observations)
    assert out.loc[out.k.eq(0),['global_scale','team_scale']].eq(1).all().all()
    w=.5**(1/20)
    expected=(4*w+8*2.5)/(w+8)
    assert np.isclose(out.loc[out.k.eq(1)&out.team.eq('a'),'team_scale'].iloc[0],expected)
    assert out.loc[out.k.eq(1)&out.team.eq('new'),'team_scale'].iloc[0]==2.5
    bad=observations.copy(); bad.loc[bad.k.ge(2),'team_mse']=999.
    after=forecast_scales(entries,bad)
    pd.testing.assert_frame_equal(out.loc[out.k.le(2)],after.loc[after.k.le(2)],check_exact=True)
    assert not out.loc[out.k.eq(3),'team_scale'].equals(after.loc[after.k.eq(3),'team_scale'])
    zero=forecast_scales(entries,observations.assign(team_mse=0.))
    assert zero.loc[zero.k.gt(0),['global_scale','team_scale']].eq(1e-4).all().all()
    null=forecast_scales(entries,observations,shuffled=True)
    pd.testing.assert_frame_equal(null,forecast_scales(entries,observations,shuffled=True),check_exact=True)
    pd.testing.assert_series_equal(null.global_scale,out.global_scale,check_exact=True)


def test_relative_labels():
    x=pd.DataFrame(dict(k=[0]*6+[1]*4,y_pace=[1.,2.,3.,4.,5.,np.nan,1.,2.,3.,4.],mean_forecast=[0.,1.,1.,2.,4.,3.,0.,1.,1.,2.]))
    a=revealed_errors(x)
    assert a.loc[a.k.eq(1),'error'].isna().all()
    np.testing.assert_allclose(a.loc[a.k.eq(0),'error'].dropna(),[-.4,-.4,.6,.6,-.4])
    shifted=x.copy(); shifted.y_pace+=100.
    b=revealed_errors(shifted)
    np.testing.assert_allclose(a.error,b.error,equal_nan=True,atol=1e-12)


def test_mean_target_independence():
    df=pd.read_csv('data/results.csv').loc[lambda d:d.season.le(2020)]
    pace=pd.read_csv('data/racepace_ff1.csv').loc[lambda d:d.season.le(2020)]
    X=build_features(df,pace); F=pre_weekend_cols(X)
    X=X.merge(pace[['season','round','code','pace_gap']].rename(columns={'pace_gap':'y_pace'}),on=['season','round','code'],how='left',validate='one_to_one')
    current=X[X.season.eq(2020)&X['round'].eq(5)]
    train=X[X.k.lt(current.k.iloc[0])&X.season.ge(2019)]
    before=pace_mean(train,current,F)
    bad=current.copy(); bad[['y_pace','order','win','podium','gridn','q_gap']]=999.
    np.testing.assert_array_equal(before,pace_mean(train,bad,F))


if __name__=='__main__':
    test_math_and_cutoff(); test_relative_labels(); test_mean_target_independence()
    print('E194 scale shrinkage/fallback, source cutoff, relative labels and mean target invariance passed')
