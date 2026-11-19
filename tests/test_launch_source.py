"""Known raw launch and invalid-sample/window checks; no real source reads."""
import numpy as np
import pandas as pd
from experiments.e233_launch_source import launch,event


def test_launch():
    origin=pd.Timestamp('2019-01-01'); t=np.arange(-.6,1.51,.3)
    car=pd.DataFrame(dict(Date=origin+pd.to_timedelta(t,unit='s'),Time=pd.to_timedelta(t+10,unit='s'),Speed=[0,0,0,10,25,40,55,70]))
    result=launch(car,origin,0)
    assert result['supported'] and np.isclose(result['time20'],.5) and np.isclose(result['time40'],.9)
    for field,value in [('Time',pd.NaT),('Speed',np.nan),('Speed',-1),('Speed',0)]:
        changed=car.copy(); changed.loc[4,field]=value
        assert not launch(changed,origin,0)['supported'],(field,value)
    changed=car.copy(); changed.loc[4:,'Date']+=pd.Timedelta(seconds=1)
    assert not launch(changed,origin,0)['supported']
    assert not launch(car.iloc[:0],origin,0)['supported']
    assert not launch(car.iloc[:1],origin,0)['supported']
    assert not launch(car,origin,20)['supported']
    assert launch(pd.concat([car,car]).iloc[::-1],origin,0)==result
    changed=car.assign(Throttle=0,Brake=True,DRS=99)
    assert launch(changed,origin,0)==result
    laps=pd.DataFrame(dict(DriverNumber=[str(i) for i in range(12)],Driver=[f'D{i}' for i in range(12)],LapNumber=1,
        PitOutTime=pd.to_timedelta([None]*12),LapStartTime=pd.to_timedelta([10]*12,unit='s')))
    cars={str(i):car for i in range(12)}
    good=event(laps,cars)
    assert good['gate'] and good['supported']==12 and good['spread20']==0
    laps.loc[0,'PitOutTime']=pd.Timedelta(seconds=11)
    assert not event(laps,cars)['gate']
    print('E233 known launch, sparse/missing/gappy/window/duplicate/clock/coverage checks pass')


if __name__=='__main__':
    test_launch()
