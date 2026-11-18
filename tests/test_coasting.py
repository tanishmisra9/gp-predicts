"""E213 zero-event laps are observations; missing telemetry is not zero coasting."""
import numpy as np
import pandas as pd

from experiments.e213_coasting import lap_measurement


def test_measurement():
    origin=pd.Timestamp('2019-01-01'); t=np.arange(41)*.25
    car=pd.DataFrame(dict(Date=origin+pd.to_timedelta(t,unit='s'),Time=pd.to_timedelta(t,unit='s'),
        Speed=250.,Throttle=100.,Brake=False))
    car.loc[10:12,'Throttle']=0.; car.loc[13,'Brake']=True
    result=lap_measurement(car,origin,0,10.01)
    np.testing.assert_allclose(result,[.75,10/10.01,1])
    assert lap_measurement(car.assign(Throttle=100.),origin,0,10.01)[0]==0
    assert lap_measurement(car.iloc[:0],origin,0,10.01) is None
    for field,value in [('Time',pd.NaT),('Throttle',np.nan),('Brake',np.nan),('Speed',np.nan)]:
        bad=car.copy(); bad.loc[11,field]=value
        assert lap_measurement(bad,origin,0,10.01)[0]==0
    bad=car.copy(); bad.loc[13,'Brake']=False
    assert lap_measurement(bad,origin,0,10.01)[0]==0
    bad=car.copy(); bad.loc[13,'Speed']=199
    assert lap_measurement(bad,origin,0,10.01)[0]==0
    bad=car.copy(); bad.loc[10:12,'Date']+=pd.Timedelta(seconds=.6)
    assert lap_measurement(bad,origin,0,10.01)[0]==0
    assert lap_measurement(car,origin,0,3.25)[0]==0  # brake sample exactly at lap end excluded
    assert lap_measurement(car,origin,2.5,10.01)[0]==0  # no observed powered predecessor
    np.testing.assert_array_equal(lap_measurement(car.assign(DRS=999),origin,0,10.01),result)
    bad=car.copy(); bad.loc[:20,'Time']=pd.NaT
    assert lap_measurement(bad,origin,0,10.01) is None


if __name__=='__main__':
    test_measurement()
    print('coasting observed coverage, no-event zero, padding, channels, gaps and half-open lap bounds PASS')
