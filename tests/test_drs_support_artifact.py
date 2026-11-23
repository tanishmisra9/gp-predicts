"""Independent scalar alignment/geometry/centers and x-sorted greedy matcher."""
import bisect
import hashlib
import json
import math
from collections import defaultdict
from statistics import median
from unittest.mock import patch
import numpy as np
import pandas as pd
from experiments.e190_traffic_feasibility import cached_session
from f1.data import ROOT


def records(frame,position=False,observed=False):
    last={r.Date.value:r for r in frame.itertuples() if pd.notna(r.Date)}
    result=[last[k] for k in sorted(last)]
    if position:return [r for r in result if pd.notna(r.Time) and r.Status=="OnTrack" and math.isfinite(r.X) and math.isfinite(r.Y)]
    if observed:return [r for r in result if pd.notna(r.Time)]
    return result


def closest(times,t):
    j=bisect.bisect_left(times,t)
    choices=[k for k in (j-1,j) if 0<=k<len(times)]
    if not choices:return None
    i=min(choices,key=lambda i:(abs(times[i]-t),times[i]))
    return i if abs(times[i]-t)<=120_000_000 else None


def scale(car,pos):
    c=records(car,observed=True);p=records(pos,position=True);times=[r.Date.value for r in c];values=[]
    for a,b in zip(p,p[1:]):
        dt=b.Date.value-a.Date.value
        if not 0<dt<=500_000_000:continue
        i=closest(times,(a.Date.value+b.Date.value)//2)
        if i is None:continue
        r=c[i]
        if not math.isfinite(r.Speed) or not 180<=r.Speed<=350 or not 98<=r.Throttle<=100 or r.Brake!=False:continue
        q=math.hypot(b.X-a.X,b.Y-a.Y)/(r.Speed*(dt/1e9)/3.6)
        if math.isfinite(q) and q>0:values.append(q)
    return dict(steps=len(values),q=median(values) if len(values)>=100 else None)


def clean_laps(raw):
    kept=[];group=defaultdict(list)
    for r in raw.itertuples():
        if r.LapNumber>1 and pd.notna(r.LapTime) and pd.notna(r.Time) and pd.notna(r.LapStartTime) and pd.isna(r.PitInTime) and pd.isna(r.PitOutTime) and str(r.TrackStatus)=="1" and pd.notna(r.IsAccurate) and r.IsAccurate:
            group[r.Driver].append(r)
    for driver,rows in group.items():
        m=median(r.LapTime.total_seconds() for r in rows)
        kept.extend(r for r in rows if r.LapTime.total_seconds()<1.07*m)
    flags={(r.Driver,int(r.LapNumber)):any(other.Driver!=r.Driver and 0<=(r.LapStartTime-other.Time).total_seconds()<=1.5 for other in kept) for r in kept}
    return kept,flags


def candidates(car,pos,laps,flags,origin,driver):
    c=records(car);p=records(pos,position=True);pt=[r.Date.value for r in p]
    intervals=sorted((origin+r.LapStartTime,origin+r.Time,r) for r in laps if r.Driver==driver)
    starts=[a.value for a,b,r in intervals];result=[]
    for i in range(2,len(c)-2):
        seq=c[i-2:i+3];r=c[i]
        if not all(pd.notna(s.Time) and math.isfinite(s.Speed) and s.Speed>=180 and 98<=s.Throttle<=100 and s.Brake==False
                   and math.isfinite(s.nGear) and 4<=s.nGear<=8 and s.nGear==int(s.nGear) for s in seq):continue
        states=[1 if s.DRS in (10,12,14) else 0 if s.DRS in (0,8) else -1 for s in seq]
        if min(states)<0 or len(set(states))!=1 or len({s.nGear for s in seq})!=1:continue
        if not all(0<(b.Date-a.Date).value<=500_000_000 for a,b in zip(seq,seq[1:])):continue
        k=bisect.bisect_right(starts,r.Date.value)-1
        if k<0:continue
        a,b,l=intervals[k]
        if not a<=seq[0].Date or not seq[-1].Date<b:continue
        j=closest(pt,r.Date.value)
        if j is None or l.Compound not in ("SOFT","MEDIUM","HARD") or pd.isna(l.Stint) or pd.isna(l.TyreLife):continue
        xy=p[j]
        result.append(dict(driver=driver,native_index=i,time_ns=r.Date.value,position_index=j,position_ns=xy.Date.value,
                           lap=int(l.LapNumber),stint=float(l.Stint),compound=l.Compound,age=float(l.TyreLife),gear=int(r.nGear),
                           speed=float(r.Speed),state=states[2],close=flags[(driver,int(l.LapNumber))],x=float(xy.X),y=float(xy.Y)))
    return result


def independent_pairs(table,radius):
    groups=defaultdict(list)
    def key(r):return (r.driver,r.stint,r.compound,r.gear,r.close)
    for r in table[table.state.eq(0)].itertuples():groups[key(r)].append(r)
    ordered={k:sorted(v,key=lambda r:(r.x,r.native_index)) for k,v in groups.items()}
    xs={k:[r.x for r in v] for k,v in ordered.items()}
    used=set();pairs=set();rows=[]
    for r in table[table.state.eq(1)].sort_values(["driver","lap","native_index"]).itertuples():
        ctx=key(r);pool=ordered.get(ctx,[]);x=xs.get(ctx,[])
        low=bisect.bisect_left(x,r.x-radius);high=bisect.bisect_right(x,r.x+radius)
        cx,cy=math.floor(r.x/radius),math.floor(r.y/radius);eligible=[]
        for off in pool[low:high]:
            cell=(r.driver,r.lap,off.lap,cx,cy)
            if (r.driver,off.native_index) in used or cell in pairs or off.lap==r.lap:continue
            if abs(off.lap-r.lap)>5 or abs(off.age-r.age)>2 or abs(off.speed-r.speed)>5:continue
            distance=math.hypot(r.x-off.x,r.y-off.y)
            if distance<=radius:eligible.append((distance,abs(r.lap-off.lap),off.lap,off.native_index,off,cell))
        if not eligible:continue
        distance,_,_,_,off,cell=min(eligible,key=lambda x:x[:4])
        used.add((r.driver,off.native_index));pairs.add(cell)
        rows.append(dict(driver=r.driver,on_index=r.native_index,off_index=off.native_index,on_lap=r.lap,off_lap=off.lap,
                         cellx=cx,celly=cy,distance=distance,speed_difference=abs(r.speed-off.speed),age_difference=abs(r.age-off.age),
                         gear=r.gear,compound=r.compound,stint=r.stint,close=r.close))
    return rows


def main():
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True)
    result=json.loads((ROOT/"results/e262_summary.json").read_text())
    original=json.loads((ROOT/"results/e247_native_channel_inventory.json").read_text())
    saved_centers=pd.read_csv(ROOT/"results/e262_centers.csv",float_precision="round_trip")
    saved_pairs=pd.read_csv(ROOT/"results/e262_pairs.csv",float_precision="round_trip")
    saved_scales=pd.read_csv(ROOT/"results/e262_scales.csv",float_precision="round_trip")
    saved_support=pd.read_csv(ROOT/"results/e262_support.csv",float_precision="round_trip")
    gates=[]
    for receipt,prior in zip(result["sessions"],original["sources"]):
        year,round_=receipt["season"],receipt["round"]
        assert year==2019 and round_ in (5,12,14) and receipt["available"]
        assert receipt["requests"]==[dict(url="https://livetiming.formula1.com"+prior["path"]+"Position.z.jsonStream",status=200)]
        assert not receipt["stop_remaining"]
        with patch("requests.adapters.HTTPAdapter.send",side_effect=RuntimeError("DRS audit HTTP forbidden")):
            s=cached_session(fastf1,year,round_);raw=pd.DataFrame(s.laps)
            cars=fastf1._api.car_data(s.api_path);positions=fastf1._api.position_data(s.api_path)
            for feed,field in (("car_data","car_stream_sha256"),("position","position_stream_sha256")):
                rows=fastf1._api.fetch_page(s.api_path,feed);blob="".join(r+"\r\n" for r in rows).encode()
                assert hashlib.sha256(blob).hexdigest()==receipt[field]
        assert receipt["car_stream_sha256"]==prior["record_stream_sha256"]
        assert sum(r["numeric"] for r in prior["rows"] if r["channel"]=="45")==prior["car_updates"]
        origin=max((c.Date-c.Time).max() for c in cars.values() if c.Time.notna().any()).round("ms")
        laps,flags=clean_laps(raw);expected=[];scales=[]
        for driver,g in raw.groupby("Driver",sort=True):
            numbers=g.DriverNumber.dropna().unique();assert len(numbers)==1
            car,pos=cars.get(str(numbers[0])),positions.get(str(numbers[0]))
            assert car is not None and pos is not None
            scales.append(dict(driver=driver,**scale(car,pos)))
            expected.extend(candidates(car,pos,laps,flags,origin,driver))
        actual=saved_centers[saved_centers["round"].eq(round_)]
        expected=pd.DataFrame(expected,columns=actual.columns[:-2])
        pd.testing.assert_frame_equal(expected.reset_index(drop=True),actual.drop(columns=["season","round"]).reset_index(drop=True),check_dtype=False,atol=0,rtol=0)
        q=median(r["q"] for r in scales if r["q"] is not None)
        np.testing.assert_allclose(q,receipt["q"],atol=1e-10,rtol=1e-12)
        sc=saved_scales[saved_scales["round"].eq(round_)]
        pd.testing.assert_frame_equal(pd.DataFrame(scales),sc.drop(columns=["season","round"]).reset_index(drop=True),check_dtype=False,atol=1e-10,rtol=1e-12)
        observed=saved_pairs[saved_pairs["round"].eq(round_)]
        pairs=pd.DataFrame(independent_pairs(expected,10*q),columns=observed.columns[:-2])
        pd.testing.assert_frame_equal(pairs,observed.drop(columns=["season","round"]).reset_index(drop=True),check_dtype=False,atol=1e-8,rtol=1e-12)
        geometry=(sum(r["steps"] for r in scales)>=10000 and sum(r["q"] is not None for r in scales)>=10 and sum(r["q"] is not None and abs(r["q"]-q)<=.1*q for r in scales)>=10 and q>0)
        assert geometry==receipt["geometry_gate"] and len(expected)==receipt["centers"] and len(pairs)==receipt["pairs"]
        assert sum(r["steps"] for r in scales)==receipt["steps"]
        assert sum(r["q"] is not None for r in scales)==receipt["qualified_scale_cars"]
        assert sum(r["q"] is not None and abs(r["q"]-q)<=.1*q for r in scales)==receipt["agreeing_scale_cars"]
        support_rows=saved_support[saved_support["round"].eq(round_)]
        supported=0
        for row in support_rows.itertuples():
            g=pairs[pairs.driver.eq(row.driver)];nl=g.on_lap.nunique();nf=g.off_lap.nunique();cells=len(g[["cellx","celly"]].drop_duplicates())
            ok=len(g)>=10 and nl>=3 and nf>=3 and cells>=2
            assert (row.pairs,row.on_laps,row.off_laps,row.cells,row.supported,row.data_available)==(len(g),nl,nf,cells,ok,True)
            supported+=ok
        assert len(support_rows)==20 and supported==receipt["supported_drivers"]
        gates.append(bool(geometry and supported>=8));assert gates[-1]==receipt["gate"]
        print("E262 audited",round_,len(expected),len(pairs),supported,flush=True)
    assert result["gate"]==(sum(gates)>=2) and not result["response_measured"] and not result["winner_fit"]
    for name,digest in result["files"].items():assert hashlib.sha256((ROOT/"results"/name).read_bytes()).hexdigest()==digest
    print("E262 scalar all3 alignment/scale/cleaning/centers/x-range greedy pairs/support/rawhash/scope/gate PASS")


if __name__=="__main__":main()
