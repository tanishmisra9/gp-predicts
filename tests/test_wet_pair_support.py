import numpy as np
import pandas as pd
from experiments.e246_wet_pair_support import pairs,support


def main():
    entries=pd.DataFrame(dict(code=["AAA","BBB"],team=["T","T"]))
    rows=[]
    for lap in range(2,14):
        for driver,pos in [("AAA",1),("BBB",2)]:
            rows.append(dict(Driver=driver,LapNumber=lap,Time=pd.Timedelta(seconds=lap*120+2*(pos-1)),
                Position=pos,LapTime=pd.Timedelta(seconds=120),PitInTime=pd.NaT,PitOutTime=pd.NaT,
                TrackStatus="1",IsAccurate=True,Compound="INTERMEDIATE",TyreLife=lap))
    raw=pd.DataFrame(rows)
    p=pairs(raw,entries)
    assert len(p)==12 and p.a.eq("AAA").all() and p.b.eq("BBB").all()
    c=raw.copy();c.loc[c.Driver.eq("BBB"),"TyreLife"]+=3
    assert len(pairs(c,entries))==12
    c.loc[c.Driver.eq("BBB"),"TyreLife"]+=.001
    assert pairs(c,entries).empty
    for col,value in [("TrackStatus","12"),("IsAccurate",False),("PitInTime",pd.Timedelta(1)),("Compound","SOFT"),("TyreLife",np.nan)]:
        c=raw.copy();c.loc[c.Driver.eq("BBB"),col]=value
        assert pairs(c,entries).empty,(col,value)
    c=raw.copy();c.loc[c.Driver.eq("BBB"),"Time"]-=pd.Timedelta(seconds=.5)
    assert pairs(c,entries).empty
    assert pairs(raw.iloc[:0],entries).empty
    try:pairs(pd.concat([raw,raw.iloc[:1]]),entries)
    except AssertionError:pass
    else:raise AssertionError("duplicate accepted")
    assert not support(p.iloc[:0].assign(season=2019,round=1))[0]["gate"]
    # Support counts alone cannot fabricate a response or winner fit.
    report,table=support(p.assign(season=2019,round=1))
    assert report["supported_pairs"]==1 and table.iloc[0].half0==6 and not report["gate"]
    assert not report["response_fit"] and not report["winner_fit"]
    print("E246 wet/samecompound/age boundary/traffic/green/accurate/pit/duplicate/empty/support synthetics PASS")


if __name__=="__main__":main()
