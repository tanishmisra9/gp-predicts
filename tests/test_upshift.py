import numpy as np
import pandas as pd
from experiments.e244_upshift import shifts


def main():
    origin=pd.Timestamp("2019-01-01")
    car=pd.DataFrame(dict(Date=origin+pd.to_timedelta([0,.25,.5,.75],unit="s"),
        Time=pd.to_timedelta([0,.25,.5,.75],unit="s"),nGear=[6,6,7,7],
        RPM=[11000,11500,10000,10500],Throttle=100,Brake=False,Speed=250))
    s=shifts(car,origin)
    assert len(s)==1 and s.iloc[0].duration==11.5 and s.iloc[0].post_rpm==10000
    assert shifts(car.iloc[:0],origin).empty and shifts(car.iloc[:3],origin).empty
    for col,val in [("Time",pd.NaT),("RPM",0),("RPM",np.nan),("Throttle",97),("Throttle",101),("Brake",True),("Brake",None),("Speed",179),("nGear",5)]:
        c=car.copy(); c.loc[1,col]=val
        assert shifts(c,origin).empty,(col,val)
    c=car.copy();c.loc[2,"RPM"]=12000
    assert shifts(c,origin).empty
    c=car.copy();c.loc[3,"nGear"]=8
    assert shifts(c,origin).empty
    c=car.copy();c.loc[2:,"Date"]+=pd.Timedelta(seconds=1)
    assert shifts(c,origin).empty
    c=car.copy();c["Date"]=origin+pd.to_timedelta([0,.5,1,1.5],unit="s")
    assert len(shifts(c,origin))==1
    c["Date"]=origin+pd.to_timedelta([0,.500001,1,1.5],unit="s")
    assert shifts(c,origin).empty
    pd.testing.assert_frame_equal(shifts(car.assign(DRS=[0,8,12,99]),origin),s)
    pd.testing.assert_frame_equal(shifts(pd.concat([car,car]).iloc[::-1],origin),s)
    print("upshift observed padding/gear/throttle/brake/RPM/gap/duplicate/DRS synthetics passed")


if __name__=="__main__":main()
