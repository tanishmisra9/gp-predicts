from datetime import date
import numpy as np
import pandas as pd
from experiments.e243_development_proxy import exposure,residuals,association


def main():
    s=[dict(season=2020,round=17,reference_date="2021-01-01",event_date="2020-12-13",
            constructors=[dict(constructor="racing_point",position=10)])]
    x,r,used=exposure(s,"aston_martin",date(2021,1,1))
    assert (x,r,used)==(0.,0.,[])
    x,r,used=exposure(s,"aston_martin",date(2021,7,1))
    assert abs(x-.5/365)<1e-14 and r==.5 and used[0]["days"]==1
    x,_,_=exposure(s,"aston_martin",date(2022,7,1))
    assert abs(x-.5)<1e-14
    poisoned=s+[dict(season=2021,round=8,reference_date="2021-07-01",event_date="2021-06-27",constructors=[])]
    assert exposure(s,"aston_martin",date(2021,7,1))==exposure(poisoned,"aston_martin",date(2021,7,1))
    s[0]["constructors"].append(dict(constructor="force_india",position=9))
    try:exposure(s,"aston_martin",date(2022,7,1))
    except AssertionError:pass
    else:raise AssertionError("ambiguous lineage accepted")
    s[0]["constructors"].append(dict(constructor="aston_martin",position=1))
    assert exposure(s,"aston_martin",date(2022,7,1))[0]==-.5
    a=np.arange(10.)
    controls=np.column_stack([np.ones(10),a])
    assert np.max(abs(residuals(2*a+3,controls)))<1e-12
    d=pd.DataFrame(dict(k=np.repeat(np.arange(3),10),x=np.tile(a,3),y=-np.tile(a,3)))
    z=association(d,"x","y")
    assert z["races"]==3 and z["rows"]==30 and abs(z["correlation"]+1)<1e-12
    assert association(d.iloc[:0],"x","y")["correlation"] is None
    print("development proxy synthetics passed")


if __name__=="__main__":main()
