"""E262 fixed position-matched recorded DRS-state support, no responses."""
import hashlib
import json
from collections import defaultdict
from unittest.mock import patch
import numpy as np
import pandas as pd
from experiments.e190_traffic_feasibility import PILOT,cached_session,start_exposure
from f1.data import ROOT

CENTER=["driver","native_index","time_ns","position_index","position_ns","lap","stint","compound","age","gear","speed","state","close","x","y"]
PAIR=["driver","on_index","off_index","on_lap","off_lap","cellx","celly","distance","speed_difference","age_difference","gear","compound","stint","close"]
PROGRESS=ROOT/"results/e262_position_status.json"


def prepared(frame,position=False,observed=False):
    d=frame[frame.Date.notna()].drop_duplicates("Date",keep="last").sort_values("Date").reset_index(drop=True)
    if position:
        d=d[d.Time.notna()&d.Status.eq("OnTrack")&np.isfinite(d[["X","Y"]].to_numpy(float)).all(axis=1)]
    elif observed:d=d[d.Time.notna()]
    return d.reset_index(drop=True)


def nearest(source,queries,tolerance=120_000_000):
    source=np.asarray(source,dtype=np.int64);queries=np.asarray(queries,dtype=np.int64)
    if not len(source):return np.full(len(queries),-1,dtype=int)
    right=np.clip(np.searchsorted(source,queries),0,len(source)-1)
    left=np.maximum(right-1,0)
    chosen=np.where(np.abs(source[right]-queries)<np.abs(source[left]-queries),right,left)
    return np.where(np.abs(source[chosen]-queries)<=tolerance,chosen,-1)


def calibration(car,pos):
    c=prepared(car,observed=True);p=prepared(pos,position=True)
    if len(p)<2 or c.empty:return dict(steps=0,q=None)
    ns=p.Date.to_numpy(dtype="datetime64[ns]").astype("int64");dt=np.diff(ns)
    mid=ns[:-1]+dt//2;which=nearest(c.Date.to_numpy(dtype="datetime64[ns]").astype("int64"),mid)
    safe=np.maximum(which,0);matched=c.iloc[safe]
    speed=matched.Speed.to_numpy(float)
    good=(which>=0)&(dt>0)&(dt<=500_000_000)&np.isfinite(speed)&(speed>=180)&(speed<=350)
    good&=matched.Throttle.between(98,100).to_numpy()&matched.Brake.eq(False).to_numpy()
    distance=np.hypot(np.diff(p.X.to_numpy(float)),np.diff(p.Y.to_numpy(float)))
    denominator=speed*dt/3.6e9
    q=np.divide(distance,denominator,out=np.full(len(dt),np.nan),where=denominator>0)
    good&=np.isfinite(q)&(q>0)
    return dict(steps=int(good.sum()),q=float(np.median(q[good])) if good.sum()>=100 else None)


def centers(clean,car,pos,origin,driver):
    d=prepared(car);p=prepared(pos,position=True);n=len(d)
    if n<5 or clean.empty or p.empty:return pd.DataFrame(columns=CENTER)
    ns=d.Date.to_numpy(dtype="datetime64[ns]").astype("int64")
    state=np.where(d.DRS.isin([10,12,14]),1,np.where(d.DRS.isin([0,8]),0,-1))
    gear=d.nGear.to_numpy(float)
    valid=d.Time.notna().to_numpy()&d.Speed.ge(180).to_numpy()&np.isfinite(d.Speed.to_numpy(float))
    valid&=d.Throttle.between(98,100).to_numpy()&d.Brake.eq(False).to_numpy()&(state>=0)
    valid&=np.isfinite(gear)&(gear>=4)&(gear<=8)&(gear==np.floor(gear))
    good=np.ones(n-4,dtype=bool)
    for j in range(5):good&=valid[j:j+n-4]&(gear[j:j+n-4]==gear[2:n-2])&(state[j:j+n-4]==state[2:n-2])
    dt=np.diff(ns)
    for j in range(4):good&=(dt[j:j+n-4]>0)&(dt[j:j+n-4]<=500_000_000)
    c=clean.sort_values("LapStartTime").reset_index(drop=True)
    a=c.LapStartTime.dt.total_seconds().to_numpy();b=c.Time.dt.total_seconds().to_numpy()
    assert (b>a).all() and (a[1:]>=b[:-1]).all()
    t=(d.Date-origin).dt.total_seconds().to_numpy();ids=np.arange(2,n-2)
    lap=np.searchsorted(a,t[ids],side="right")-1;safe=np.maximum(lap,0)
    good&=(lap>=0)&(t[ids-2]>=a[safe])&(t[ids+2]<b[safe])
    which=nearest(p.Date.to_numpy(dtype="datetime64[ns]").astype("int64"),ns[ids])
    good&=which>=0
    ids=ids[good];lap=lap[good];which=which[good]
    if not len(ids):return pd.DataFrame(columns=CENTER)
    l=c.iloc[lap];loc=p.iloc[which]
    output=pd.DataFrame(dict(driver=driver,native_index=ids,time_ns=ns[ids],position_index=which,
        position_ns=loc.Date.to_numpy(dtype="datetime64[ns]").astype("int64"),lap=l.LapNumber.to_numpy(int),
        stint=l.Stint.to_numpy(float),compound=l.Compound.to_numpy(),age=l.TyreLife.to_numpy(float),
        gear=gear[ids].astype(int),speed=d.Speed.to_numpy(float)[ids],state=state[ids],close=l.close.to_numpy(bool),
        x=loc.X.to_numpy(float),y=loc.Y.to_numpy(float)))
    return output[output.compound.isin(["SOFT","MEDIUM","HARD"])&np.isfinite(output[["stint","age"]].to_numpy()).all(axis=1)][CENTER]


def match(table,radius):
    assert np.isfinite(radius) and radius>0 and not table.duplicated(["driver","native_index"]).any()
    buckets=defaultdict(list)
    def context(r):return (r.driver,r.stint,r.compound,r.gear,r.close)
    for r in table[table.state.eq(0)].itertuples():
        buckets[context(r)+(int(np.floor(r.x/radius)),int(np.floor(r.y/radius)))].append(r)
    used=set();cells=set();rows=[]
    for r in table[table.state.eq(1)].sort_values(["driver","lap","native_index"]).itertuples():
        cx,cy=int(np.floor(r.x/radius)),int(np.floor(r.y/radius));candidates=[]
        for x in range(cx-1,cx+2):
            for y in range(cy-1,cy+2):
                for off in buckets.get(context(r)+(x,y),[]):
                    cell=(r.driver,r.lap,off.lap,cx,cy)
                    if (off.driver,off.native_index) in used or cell in cells:continue
                    if not 0<abs(r.lap-off.lap)<=5 or abs(r.age-off.age)>2 or abs(r.speed-off.speed)>5:continue
                    distance=float(np.hypot(r.x-off.x,r.y-off.y))
                    if distance<=radius:candidates.append((distance,abs(r.lap-off.lap),off.lap,off.native_index,off,cell))
        if not candidates:continue
        distance,_,_,_,off,cell=min(candidates,key=lambda x:x[:4])
        used.add((off.driver,off.native_index));cells.add(cell)
        rows.append(dict(driver=r.driver,on_index=r.native_index,off_index=off.native_index,on_lap=r.lap,off_lap=off.lap,
            cellx=cx,celly=cy,distance=distance,speed_difference=abs(r.speed-off.speed),age_difference=abs(r.age-off.age),
            gear=r.gear,compound=r.compound,stint=r.stint,close=r.close))
    return pd.DataFrame(rows,columns=PAIR)


def support(pairs,drivers):
    rows=[]
    for driver in sorted(drivers):
        g=pairs[pairs.driver.eq(driver)]
        on=int(g.on_lap.nunique());off=int(g.off_lap.nunique());cells=len(g[["cellx","celly"]].drop_duplicates())
        rows.append(dict(driver=driver,pairs=len(g),on_laps=on,off_laps=off,cells=cells,supported=bool(len(g)>=10 and on>=3 and off>=3 and cells>=2)))
    return rows


def urls(path):
    assert path.startswith("/static/2019/") and path.endswith("_Race/")
    return {host+path+"Position.z.jsonStream" for host in ("https://livetiming.formula1.com","https://livetiming-mirror.fastf1.dev")}


def bounded_send(send,allowed,audit,stop):
    def wrapped(adapter,request,**kwargs):
        if stop["denied"] or request.url not in allowed:raise RuntimeError("E262 endpoint forbidden")
        try:response=send(adapter,request,**kwargs)
        except Exception as exc:
            audit.append(dict(url=request.url,error_type=type(exc).__name__));raise
        audit.append(dict(url=request.url,status=int(response.status_code)))
        if response.status_code in (401,403,429):stop["denied"]=True
        return response
    return wrapped


def acquire():
    import fastf1
    import requests
    import f1.ff1
    from fastf1.internals.f1account import get_token
    reports=json.loads(PROGRESS.read_text()) if PROGRESS.exists() else []
    done={(r["season"],r["round"]) for r in reports}
    assert len(done)==len(reports) and done.issubset(set(PILOT))
    stop={"denied":any(r.get("stop_remaining",False) or any(x.get("status") in (401,403,429) for x in r["requests"]) for r in reports)}
    authenticated=False
    for season,round_ in PILOT:
        if (season,round_) in done:continue
        report=dict(season=season,round=round_,available=False,requests=[])
        if stop["denied"]:report["reason"]="unattempted_after_denial"
        else:
            fastf1.Cache.offline_mode(True)
            session=cached_session(fastf1,season,round_)
            try:
                with patch("requests.adapters.HTTPAdapter.send",side_effect=RuntimeError("cache probe HTTP forbidden")):
                    positions=fastf1._api.position_data(session.api_path)
            except Exception:
                fastf1.Cache.offline_mode(False)
                if not authenticated:
                    try:get_token(method="password")
                    except Exception as exc:
                        report["error_type"]=type(exc).__name__;stop["denied"]=True
                    else:authenticated=True
                if authenticated:
                    try:
                        with patch("requests.adapters.HTTPAdapter.send",new=bounded_send(requests.adapters.HTTPAdapter.send,urls(session.api_path),report["requests"],stop)):
                            positions=fastf1._api.position_data(session.api_path)
                    except Exception as exc:report["error_type"]=type(exc).__name__
                    else:report["available"]=bool(positions)
            else:report["available"]=bool(positions)
        report["stop_remaining"]=stop["denied"]
        reports.append(report);PROGRESS.write_text(json.dumps(reports,indent=2)+"\n")
    return reports


def main():
    import fastf1
    import f1.ff1
    reports=acquire();fastf1.Cache.offline_mode(True)
    inventory=json.loads((ROOT/"results/e247_native_channel_inventory.json").read_text())
    all_centers=[];all_pairs=[];all_support=[];all_scales=[];results=[]
    for report,receipt in zip(reports,inventory["sources"]):
        season,round_=report["season"],report["round"];assert (season,round_)==(receipt["season"],receipt["round"])
        result=dict(report)
        if report["available"]:
            with patch("requests.adapters.HTTPAdapter.send",side_effect=RuntimeError("E262 measurement HTTP forbidden")):
                s=cached_session(fastf1,season,round_);laps=pd.DataFrame(s.laps)
                cars=fastf1._api.car_data(s.api_path);positions=fastf1._api.position_data(s.api_path)
                records=fastf1._api.fetch_page(s.api_path,"car_data")
                position_records=fastf1._api.fetch_page(s.api_path,"position")
            def digest(records):
                h=hashlib.sha256()
                for line in records:h.update(line.encode());h.update(b"\r\n")
                return h.hexdigest()
            assert digest(records)==receipt["record_stream_sha256"]
            assert sum(r["numeric"] for r in receipt["rows"] if r["channel"]=="45")==receipt["car_updates"]
            origin=max((c.Date-c.Time).max() for c in cars.values() if c.Time.notna().any()).round("ms")
            clean=start_exposure(laps);parts=[];scales=[]
            drivers=sorted(laps.Driver.dropna().unique())
            for driver,g in laps.groupby("Driver",sort=True):
                numbers=g.DriverNumber.dropna().unique();assert len(numbers)==1
                car=cars.get(str(numbers[0]));pos=positions.get(str(numbers[0]))
                if car is None or pos is None:scales.append(dict(driver=driver,steps=0,q=None));continue
                scales.append(dict(driver=driver,**calibration(car,pos)))
                parts.append(centers(clean[clean.Driver.eq(driver)],car,pos,origin,driver))
            scale=pd.DataFrame(scales);values=scale.q.dropna();q=float(values.median()) if len(values) else None
            agreeing=int((values.sub(q).abs()<=.1*q).sum()) if q is not None else 0
            geometry=bool(scale.steps.sum()>=10000 and len(values)>=10 and agreeing>=10 and q>0)
            table=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=CENTER)
            pairs=match(table,10*q) if q is not None and q>0 else pd.DataFrame(columns=PAIR)
            supported=support(pairs,drivers)
            number_for={driver:str(g.DriverNumber.dropna().iloc[0]) for driver,g in laps.groupby("Driver")}
            for row in supported:row["data_available"]=number_for[row["driver"]] in cars and number_for[row["driver"]] in positions
            result.update(car_stream_sha256=digest(records),position_stream_sha256=digest(position_records),
                q=q,steps=int(scale.steps.sum()),qualified_scale_cars=len(values),agreeing_scale_cars=agreeing,geometry_gate=geometry,
                centers=len(table),pairs=len(pairs),supported_drivers=sum(r["supported"] for r in supported),
                gate=bool(geometry and sum(r["supported"] for r in supported)>=8))
            all_centers.append(table.assign(season=season,round=round_));all_pairs.append(pairs.assign(season=season,round=round_))
            all_support.extend(dict(r,season=season,round=round_) for r in supported);all_scales.append(scale.assign(season=season,round=round_))
        else:result["gate"]=False
        results.append(result)
    files={}
    frames={"centers":pd.concat(all_centers,ignore_index=True) if all_centers else pd.DataFrame(columns=CENTER+["season","round"]),
            "pairs":pd.concat(all_pairs,ignore_index=True) if all_pairs else pd.DataFrame(columns=PAIR+["season","round"]),
            "support":pd.DataFrame(all_support,columns=["driver","pairs","on_laps","off_laps","cells","supported","data_available","season","round"]),"scales":pd.concat(all_scales,ignore_index=True) if all_scales else pd.DataFrame(columns=["driver","steps","q","season","round"])}
    for name,frame in frames.items():
        path=ROOT/f"results/e262_{name}.csv";frame.to_csv(path,index=False);files[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    result=dict(sessions=results,gate=bool(all(r["available"] for r in reports) and sum(r["gate"] for r in results)>=2),
                response_measured=False,winner_fit=False,files=files)
    (ROOT/"results/e262_summary.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":main()
