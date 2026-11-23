"""E263 fixed DRS-coded contrast, lap-connected halves; no forecast."""
import hashlib
import json
from collections import defaultdict
from unittest.mock import patch
import numpy as np
import pandas as pd
from experiments.e190_traffic_feasibility import cached_session
from experiments.e262_drs_support import prepared
from f1.data import ROOT

KEYS=["season","round","driver"]


def acceleration(frame,index,state):
    assert state in (0,1) and 2<=index<len(frame)-2
    w=frame.iloc[index-2:index+3]
    ns=w.Date.to_numpy(dtype="datetime64[ns]").astype("int64")
    dt=np.diff(ns)
    speed=w.Speed.to_numpy(float)
    assert w.Time.notna().all() and np.isfinite(speed).all() and (speed>=180).all()
    assert ((dt>0)&(dt<=500_000_000)).all()
    assert w.Throttle.between(98,100).all() and w.Brake.eq(False).all()
    gear=w.nGear.to_numpy(float)
    assert np.isfinite(gear).all() and ((gear>=4)&(gear<=8)&(gear==np.floor(gear))).all() and len(set(gear))==1
    assert w.DRS.isin([10,12,14] if state else [0,8]).all()
    t=(ns-ns[2])/1e9;v=speed/3.6
    centered=t-t.mean()
    a=float(centered@(v-v.mean())/(centered@centered))
    return a,float((ns[-1]-ns[0])/1e9)


def lap_components(pairs):
    graph=defaultdict(set)
    for r in pairs.itertuples():
        assert float(r.on_lap).is_integer() and float(r.off_lap).is_integer() and r.on_lap!=r.off_lap
        a,b=int(r.on_lap),int(r.off_lap);graph[a].add(b);graph[b].add(a)
    remaining=set(graph);mapping={};number=0
    while remaining:
        seed=min(remaining);todo=[seed];seen=set()
        while todo:
            current=todo.pop()
            if current in seen:continue
            seen.add(current);todo.extend(graph[current]-seen)
        for lap in seen:mapping[lap]=number
        remaining-=seen;number+=1
    return np.array([mapping[int(r.on_lap)] for r in pairs.itertuples()],dtype=int)


def aggregate(pairs,support):
    assert not pairs.duplicated(KEYS+["on_index","off_index"]).any()
    assert not support.duplicated(KEYS).any()
    assert np.isfinite(pairs.contrast.to_numpy(float)).all()
    assert set(map(tuple,pairs[KEYS].drop_duplicates().to_numpy())).issubset(set(map(tuple,support[KEYS].to_numpy())))
    pieces=[]
    for _,g in pairs.groupby(KEYS,sort=True):
        g=g.copy();g["component"]=lap_components(g);g["half"]=g.component.mod(2);pieces.append(g)
    tagged=pd.concat(pieces,ignore_index=True) if pieces else pairs.assign(component=pd.Series(dtype=int),half=pd.Series(dtype=int))
    lp=tagged.groupby(KEYS+["component","half","on_lap","off_lap"],sort=True).contrast.agg(["mean","size"]).reset_index().rename(columns={"mean":"effect","size":"cells"})
    components=lp.groupby(KEYS+["component","half"],sort=True).effect.mean().reset_index()
    rows=[]
    for r in support.sort_values(KEYS).itertuples():
        g=tagged[tagged.season.eq(r.season)&tagged["round"].eq(r.round)&tagged.driver.eq(r.driver)]
        c=components[components.season.eq(r.season)&components["round"].eq(r.round)&components.driver.eq(r.driver)]
        row=dict(season=r.season,round=r.round,driver=r.driver,original_supported=bool(r.supported),pairs=len(g),components=len(c),full=float(c.effect.mean()) if len(c) else np.nan)
        ok=bool(r.supported)
        for h in (0,1):
            half=g[g.half.eq(h)];units=c[c.half.eq(h)]
            row[f"half{h}"]=float(units.effect.mean()) if len(units) else np.nan
            row[f"pairs{h}"]=len(half);row[f"on_laps{h}"]=int(half.on_lap.nunique());row[f"off_laps{h}"]=int(half.off_lap.nunique())
            row[f"lap_pairs{h}"]=len(half[["on_lap","off_lap"]].drop_duplicates())
            ok&=len(half)>=5 and row[f"on_laps{h}"]>=2 and row[f"off_laps{h}"]>=2 and row[f"lap_pairs{h}"]>=2
        row["repeat_supported"]=bool(ok);rows.append(row)
    return tagged,lp,components,pd.DataFrame(rows)


def correlation(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<2 or not np.isfinite(x).all() or not np.isfinite(y).all() or x.std()==0 or y.std()==0:return None
    return float(np.corrcoef(x,y)[0,1])


def evaluate(table):
    pilots=[];eligible=[]
    for (year,round_),g in table.groupby(["season","round"],sort=True):
        g=g[g.repeat_supported].sort_values("driver")
        r=correlation(g.half0,g.half1)
        pilots.append(dict(season=int(year),round=int(round_),supported=len(g),correlation=r,qualified=bool(len(g)>=6)))
        if len(g)>=6:
            a=g.copy();a["x"]=a.half0-a.half0.mean();a["y"]=a.half1-a.half1.mean();eligible.append(a)
    result=dict(pilots=pilots,coverage_gate=len(eligible)>=2,gate=False,bootstrap_finite=0,bootstrap_ci=None,
                pooled_correlation=None,identity_null_tail=None,identity_null_finite=0)
    if len(eligible)<2:
        result["reason"]="insufficient_lap_disjoint_support";return result,pd.DataFrame()
    scored=pd.concat(eligible,ignore_index=True);r=correlation(scored.x,scored.y)
    rng=np.random.default_rng(263);boots=[]
    for indices in rng.integers(len(eligible),size=(4000,len(eligible))):
        draw=pd.concat([eligible[int(i)] for i in indices],ignore_index=True)
        boots.append(correlation(draw.x,draw.y))
    finite=[v for v in boots if v is not None and np.isfinite(v)]
    ci=np.quantile(finite,[.025,.975]).tolist() if len(finite)==4000 else None
    rng=np.random.default_rng(263);null=[]
    x=scored.x.to_numpy()
    for _ in range(4000):
        y=np.concatenate([g.y.to_numpy()[rng.permutation(len(g))] for g in eligible])
        null.append(correlation(x,y))
    null_finite=[v for v in null if v is not None and np.isfinite(v)]
    tail=(1+sum(v>=r for v in null_finite))/4001 if r is not None and len(null_finite)==4000 else None
    result.update(pooled_correlation=r,bootstrap_finite=len(finite),bootstrap_ci=ci,identity_null_tail=tail,identity_null_finite=len(null_finite))
    result["gate"]=bool(r is not None and r>=.4 and sum(p["qualified"] and p["correlation"] is not None and p["correlation"]>=.4 for p in pilots)>=2 and ci is not None and ci[0]>0)
    result["reason"]="repeatability_pass" if result["gate"] else "repeatability_failed"
    return result,scored


def main():
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True)
    previous=json.loads((ROOT/"results/e262_summary.json").read_text());assert previous["gate"]
    assert [(r["season"],r["round"]) for r in previous["sessions"]]==[(2019,5),(2019,12),(2019,14)]
    for name,digest in previous["files"].items():assert hashlib.sha256((ROOT/"results"/name).read_bytes()).hexdigest()==digest
    pairs=pd.read_csv(ROOT/"results/e262_pairs.csv",float_precision="round_trip")
    centers=pd.read_csv(ROOT/"results/e262_centers.csv",float_precision="round_trip").set_index(KEYS+["native_index"],verify_integrity=True)
    support=pd.read_csv(ROOT/"results/e262_support.csv",float_precision="round_trip")
    assert len(pairs)==813 and len(support)==60
    measured=[];hashes=[]
    for receipt in previous["sessions"]:
        year,round_=receipt["season"],receipt["round"]
        with patch("requests.adapters.HTTPAdapter.send",side_effect=RuntimeError("E263 HTTP forbidden")):
            s=cached_session(fastf1,year,round_);laps=pd.DataFrame(s.laps);cars=fastf1._api.car_data(s.api_path)
            records=fastf1._api.fetch_page(s.api_path,"car_data")
        digest=hashlib.sha256()
        for line in records:digest.update(line.encode());digest.update(b"\r\n")
        assert digest.hexdigest()==receipt["car_stream_sha256"];hashes.append(digest.hexdigest())
        canonical={str(k):prepared(v) for k,v in cars.items()}
        for r in pairs[pairs["round"].eq(round_)].itertuples():
            numbers=laps[laps.Driver.eq(r.driver)].DriverNumber.dropna().unique();assert len(numbers)==1
            car=canonical[str(numbers[0])];row=r._asdict();row.pop("Index",None)
            for name,state in (("on",1),("off",0)):
                i=int(getattr(r,name+"_index"));center=centers.loc[(year,round_,r.driver,i)]
                assert car.Date.iloc[i].value==center.time_ns and center.state==state
                a,span=acceleration(car,i,state);row[name+"_acceleration"]=a;row[name+"_span"]=span
            row["contrast"]=row["on_acceleration"]-row["off_acceleration"];measured.append(row)
    measured=pd.DataFrame(measured).sort_values(KEYS+["on_index","off_index"]).reset_index(drop=True)
    tagged,lp,components,table=aggregate(measured,support);result,scored=evaluate(table)
    files={}
    for name,frame in (("pairs",tagged),("lap_pairs",lp),("components",components),("measurements",table),("scored",scored)):
        path=ROOT/f"results/e263_{name}.csv";frame.to_csv(path,index=False);files[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    result.update(pairs=len(measured),source_car_hashes=hashes,files=files,winner_fit=False,temporal_forecast=False,physical_causality_verified=False)
    (ROOT/"results/e263_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
