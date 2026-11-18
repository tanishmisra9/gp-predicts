"""Forecast admission and strict-past actual-weather/label histories."""
import numpy as np
import pandas as pd
from experiments.e224_weather_winner import NEW,add_weather


def test_timing():
    X=pd.DataFrame([dict(season=2019,round=r,k=r-1,code=c,order=o,d10_order=10.)
                    for r in (1,2,3) for c,o in [('A',1.),('B',19.)]])
    calendar=pd.DataFrame(dict(season=[2019]*3,round=[1,2,3],date=['2019-03-17','2019-03-24','2019-03-31']))
    actual=pd.DataFrame(dict(season=[2019]*3,round=[1,2,3],rain_act=[5.,0.,5.],temp_act=[30.,20.,30.]))
    forecasts=pd.DataFrame([dict(season=2019,round=r,issue=d+'T12:00:00Z',available=d+'T17:00:00Z',forecast_rain=1.,forecast_temp=30.)
                            for r,d in [(1,'2019-03-13'),(2,'2019-03-20'),(3,'2019-03-27')]])
    original=add_weather(X,actual,forecasts,calendar)
    assert original.loc[X.k.eq(0),NEW].eq(0).all().all()
    assert original.loc[X.k.eq(1),NEW].abs().sum().sum()>0
    a=actual.copy(); a.loc[a['round'].ge(2),['rain_act','temp_act']]=[999.,999.]
    x=X.copy(); x.loc[x.k.ge(1),'order']=777.
    changed=add_weather(x,a,forecasts,calendar)
    np.testing.assert_array_equal(original.loc[X.k.le(1),NEW],changed.loc[X.k.le(1),NEW])
    a=actual.copy(); a.loc[a['round'].eq(1),'rain_act']=0.
    assert not original.loc[X.k.eq(1),NEW[0]].equals(add_weather(X,a,forecasts,calendar).loc[X.k.eq(1),NEW[0]])
    late=forecasts.copy(); late['available']='2026-11-18T00:00:00Z'; late[['forecast_temp','forecast_rain']]=999.
    extended=add_weather(X,actual,pd.concat([forecasts,late],ignore_index=True),calendar)
    np.testing.assert_array_equal(original[NEW],extended[NEW])
    try:add_weather(X,actual,late,calendar)
    except AssertionError:pass
    else:raise AssertionError('unavailable target forecasts admitted')
    altered=forecasts.copy(); altered.loc[altered['round'].eq(2),'forecast_rain']=0.
    assert not original.loc[X.k.eq(1),NEW[0]].equals(add_weather(X,actual,altered,calendar).loc[X.k.eq(1),NEW[0]])


if __name__=='__main__':
    test_timing()
    print('Forecast winner source timing tests ok')
