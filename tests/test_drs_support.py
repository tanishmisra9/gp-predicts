import numpy as np
import pandas as pd
from types import SimpleNamespace
from experiments.e262_drs_support import nearest,calibration,centers,match,support,urls,bounded_send,CENTER


def fixture():
    origin=pd.Timestamp("2019-01-01")
    t=np.arange(150)*.24
    car=pd.DataFrame(dict(Date=origin+pd.to_timedelta(t+.12,unit="s"),Time=pd.to_timedelta(t+.12,unit="s"),Speed=250.,nGear=8,DRS=12,Throttle=100,Brake=False))
    pos=pd.DataFrame(dict(Date=origin+pd.to_timedelta(t,unit="s"),Time=pd.to_timedelta(t,unit="s"),Status="OnTrack",X=t*250/3.6*10,Y=0))
    laps=pd.DataFrame(dict(LapStartTime=pd.to_timedelta([0],unit="s"),Time=pd.to_timedelta([60],unit="s"),LapNumber=[2],Stint=[1],Compound=["MEDIUM"],TyreLife=[5],close=[False]))
    return origin,car,pos,laps


def main():
    np.testing.assert_array_equal(nearest([0,240_000_000],[120_000_000,360_000_000,360_000_001]),[0,1,-1])
    np.testing.assert_array_equal(nearest([], [0]),[-1])
    origin,car,pos,laps=fixture()
    c=calibration(car,pos);assert c["steps"]==149 and abs(c["q"]-10)<1e-10
    assert calibration(car.iloc[:0],pos)==dict(steps=0,q=None)
    assert calibration(car.assign(Time=pd.NaT),pos)==dict(steps=0,q=None)
    assert calibration(car,pos.assign(Status="OffTrack"))==dict(steps=0,q=None)
    assert calibration(car,pos.assign(X=0))["q"] is None
    found=centers(laps,car,pos,origin,"AAA")
    assert len(found)==146 and found.state.eq(1).all()
    for column,value in [("DRS",9),("nGear",3),("Time",pd.NaT),("Throttle",97),("Brake",True),("Speed",179)]:
        bad=car.copy();bad[column]=value
        assert centers(laps,bad,pos,origin,"AAA").empty,(column,value)
    assert centers(laps,car,pos.assign(Time=pd.NaT),origin,"AAA").empty
    assert centers(laps,car,pos.assign(Status="OffTrack"),origin,"AAA").empty
    assert centers(laps.assign(Compound="INTERMEDIATE"),car,pos,origin,"AAA").empty
    assert centers(laps,car.iloc[:4],pos,origin,"AAA").empty
    broken=car.copy();broken.loc[70:,"Date"]+=pd.Timedelta(seconds=1)
    broken_centers=centers(laps,broken,pos,origin,"AAA")
    assert not broken_centers.native_index.between(68,71).any()
    off=found.iloc[:1].copy();off["state"]=0;off["native_index"]=1;off["lap"]=3;off["age"]=7
    on=found.iloc[:1].copy();on["native_index"]=100;on["x"]=off.x.iloc[0]+10;on["speed"]=255
    pair=match(pd.concat([off,on]),10)
    assert len(pair)==1 and pair.iloc[0].distance==10 and pair.iloc[0].speed_difference==5 and pair.iloc[0].age_difference==2
    for col,val in [("x",off.x.iloc[0]+10.0001),("speed",255.001),("age",9.001),("lap",9),("gear",7),("compound","SOFT"),("stint",2),("close",True)]:
        edited=on.copy();edited[col]=val
        assert match(pd.concat([off,edited]),10).empty,(col,val)
    second=on.copy();second["native_index"]=101
    assert len(match(pd.concat([off,on,second]),10))==1
    assert match(pd.DataFrame(columns=CENTER),10).empty
    g=pd.concat([pair.assign(driver="AAA",on_lap=2+i%3,off_lap=5+i%3,cellx=i%2) for i in range(10)])
    assert support(g,["AAA"])[0]["supported"]
    assert not support(g.iloc[:9],["AAA"])[0]["supported"]
    allowed=urls("/static/2019/pilot_Race/");audit=[];stop={"denied":False};calls=[]
    def send(adapter,req,**kwargs):calls.append(req.url);return SimpleNamespace(status_code=403)
    wrapped=bounded_send(send,allowed,audit,stop);request=SimpleNamespace(url=sorted(allowed)[0])
    assert wrapped(None,request).status_code==403
    try:wrapped(None,request)
    except RuntimeError:pass
    else:raise AssertionError("denial did not stop")
    assert len(calls)==len(audit)==1 and stop["denied"]
    try:bounded_send(send,allowed,[],{"denied":False})(None,SimpleNamespace(url="https://example.com"))
    except RuntimeError:pass
    else:raise AssertionError("off-scope URL accepted")
    for path in ("/static/2026/pilot_Race/","/static/2019/pilot_Qualifying/"):
        try:urls(path)
        except AssertionError:pass
        else:raise AssertionError("off-scope path accepted")
    print("E262 nearest/calibration/centers/matching/boundaries/no-reuse/support/endpoint/denial tests PASS")


if __name__=="__main__":main()
