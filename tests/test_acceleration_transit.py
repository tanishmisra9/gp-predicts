"""Synthetic raw-channel boundaries for E232; no real telemetry read."""
import numpy as np
import pandas as pd
from experiments.e232_acceleration_transit import acceleration


def test_transit():
    origin=pd.Timestamp('2019-01-01')
    t=np.arange(6)*.3
    car=pd.DataFrame(dict(Date=origin+pd.to_timedelta(t,unit='s'),Time=pd.to_timedelta(t,unit='s'),
        Speed=[90,105,120,135,150,160],Throttle=100,Brake=False))
    row=acceleration(car,origin).iloc[0]
    assert row.samples==5 and np.isclose(row.start,.2) and np.isclose(row.end,1.2) and np.isclose(row.duration,1)
    assert acceleration(car.iloc[:0],origin).empty and acceleration(car.iloc[:1],origin).empty
    for field,value in [('Time',pd.NaT),('Throttle',97),('Throttle',101),('Throttle',np.nan),('Brake',True),('Brake',None),('Speed',np.nan),('Speed',102)]:
        changed=car.copy(); changed.loc[2,field]=value
        assert acceleration(changed,origin).empty,(field,value)
    changed=car.copy(); changed.loc[2:,'Date']+=pd.Timedelta(seconds=1)
    assert acceleration(changed,origin).empty
    changed=car.copy(); changed['DRS']=[0,12,14,8,99,0]
    pd.testing.assert_frame_equal(acceleration(changed,origin),acceleration(car,origin))
    pd.testing.assert_frame_equal(acceleration(pd.concat([car,car]).iloc[::-1],origin),acceleration(car,origin))
    changed=car.copy(); changed['Date']=origin+pd.to_timedelta(t*5,unit='s')
    assert acceleration(changed,origin).empty
    from experiments.e211_braking_signature import team_table
    empty=acceleration(car.iloc[:0],origin).assign(season=2019,round=1,driver='AAA',lap=2)
    entries=pd.DataFrame(dict(season=[2019,2019],round=[1,1],team=['T','T'],code=['AAA','BBB']))
    table=team_table(empty,entries)
    assert len(table)==1 and table.response.isna().all()
    print('E232 synthetic observed/full-throttle/brake/gap/monotone/empty/duplicate/DRS checks pass')


if __name__=='__main__':
    test_transit()
