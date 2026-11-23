"""Independent SVD windows, disjoint-set laps, scalar weights and gate audit."""
import hashlib
import json
import math
from collections import defaultdict
from unittest.mock import patch
import numpy as np
import pandas as pd
from experiments.e190_traffic_feasibility import cached_session
from f1.data import ROOT

KEYS=["season","round","driver"]


def pearson(x,y):
    if len(x)<2:return None
    mx=math.fsum(x)/len(x);my=math.fsum(y)/len(y)
    dx=[v-mx for v in x];dy=[v-my for v in y]
    vx=math.fsum(v*v for v in dx);vy=math.fsum(v*v for v in dy)
    return math.fsum(a*b for a,b in zip(dx,dy))/math.sqrt(vx*vy) if vx>0 and vy>0 else None


def svd_window(records,index,state):
    seq=records[index-2:index+3];assert len(seq)==5
    center=records[index].Date.value
    t=np.array([(r.Date.value-center)/1e9 for r in seq])
    v=np.array([r.Speed/3.6 for r in seq])
    assert all(pd.notna(r.Time) and math.isfinite(r.Speed) and r.Speed>=180 and 98<=r.Throttle<=100 and r.Brake==False for r in seq)
    assert len({r.nGear for r in seq})==1 and all(4<=r.nGear<=8 and int(r.nGear)==r.nGear for r in seq)
    assert all(0<(b.Date-a.Date).value<=500_000_000 for a,b in zip(seq,seq[1:]))
    assert all(r.DRS in ((10,12,14) if state else (0,8)) for r in seq)
    coefficients,_,rank,_=np.linalg.lstsq(np.column_stack((t,np.ones(5))),v,rcond=None)
    assert rank==2
    return float(coefficients[0]),float(t[-1]-t[0])


def main():
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True)
    old=json.loads((ROOT/"results/e262_summary.json").read_text())
    result=json.loads((ROOT/"results/e263_summary.json").read_text())
    for name,digest in old["files"].items():assert hashlib.sha256((ROOT/"results"/name).read_bytes()).hexdigest()==digest
    frozen=pd.read_csv(ROOT/"results/e262_pairs.csv",float_precision="round_trip")
    saved=pd.read_csv(ROOT/"results/e263_pairs.csv",float_precision="round_trip")
    table=pd.read_csv(ROOT/"results/e263_measurements.csv",float_precision="round_trip")
    lp=pd.read_csv(ROOT/"results/e263_lap_pairs.csv",float_precision="round_trip")
    components=pd.read_csv(ROOT/"results/e263_components.csv",float_precision="round_trip")
    support=pd.read_csv(ROOT/"results/e262_support.csv",float_precision="round_trip")
    assert len(saved)==len(frozen)==813 and len(table)==len(support)==60
    order=KEYS+["on_index","off_index"]
    pd.testing.assert_frame_equal(saved[frozen.columns].sort_values(order).reset_index(drop=True),frozen.sort_values(order).reset_index(drop=True),check_dtype=False,atol=0,rtol=0)
    values={};window_count=0
    for receipt in old["sessions"]:
        year,round_=receipt["season"],receipt["round"];assert year==2019 and round_ in (5,12,14)
        with patch("requests.adapters.HTTPAdapter.send",side_effect=RuntimeError("E263 audit HTTP forbidden")):
            s=cached_session(fastf1,year,round_);laps=pd.DataFrame(s.laps);cars=fastf1._api.car_data(s.api_path)
            raw=fastf1._api.fetch_page(s.api_path,"car_data")
        digest=hashlib.sha256("".join(line+"\r\n" for line in raw).encode()).hexdigest()
        assert digest==receipt["car_stream_sha256"]==result["source_car_hashes"][len(values)]
        values[round_]=digest
        for driver,g in saved[saved["round"].eq(round_)].groupby("driver",sort=True):
            numbers=laps[laps.Driver.eq(driver)].DriverNumber.dropna().unique();assert len(numbers)==1
            unique={r.Date.value:r for r in cars[str(numbers[0])].itertuples() if pd.notna(r.Date)}
            records=[unique[t] for t in sorted(unique)]
            for row in g.itertuples():
                a,span_a=svd_window(records,int(row.on_index),1)
                b,span_b=svd_window(records,int(row.off_index),0)
                np.testing.assert_allclose([row.on_acceleration,row.off_acceleration,row.contrast,row.on_span,row.off_span],[a,b,a-b,span_a,span_b],atol=1e-10,rtol=1e-11)
                window_count+=2
        print("E263 audited native SVD",round_,flush=True)
    assert window_count==1626
    unit_values={};component_keys={};support_map={tuple(r[k] for k in KEYS):r for _,r in support.iterrows()}
    for key,g in saved.groupby(KEYS,sort=True):
        parent={lap:lap for r in g.itertuples() for lap in (int(r.on_lap),int(r.off_lap))}
        def root(lap):
            while parent[lap]!=lap:lap=parent[lap]
            return lap
        for r in g.itertuples():parent[root(int(r.on_lap))]=root(int(r.off_lap))
        clusters=defaultdict(set)
        for lap in parent:clusters[root(lap)].add(lap)
        ordered=sorted(clusters.values(),key=min)
        mapping={lap:i for i,cluster in enumerate(ordered) for lap in cluster}
        frames={0:set(),1:set()};lap_sets={0:set(),1:set()}
        for row in g.itertuples():
            c=mapping[int(row.on_lap)];assert c==mapping[int(row.off_lap)]==row.component and row.half==c%2
            lap_sets[row.half].update((row.on_lap,row.off_lap))
            frames[row.half].update(range(row.on_index-2,row.on_index+3));frames[row.half].update(range(row.off_index-2,row.off_index+3))
            unit=key+(c,c%2,int(row.on_lap),int(row.off_lap))
            unit_values.setdefault(unit,[]).append(row.contrast)
        assert not lap_sets[0]&lap_sets[1] and not frames[0]&frames[1]
        component_keys[key]=ordered
    expected_lp=[];ce=defaultdict(list)
    for key,vals in sorted(unit_values.items()):
        effect=math.fsum(vals)/len(vals);expected_lp.append(dict(zip(KEYS+["component","half","on_lap","off_lap","effect","cells"],key+(effect,len(vals)))))
        ce[key[:5]].append(effect)
    pd.testing.assert_frame_equal(pd.DataFrame(expected_lp),lp,check_dtype=False,atol=1e-10,rtol=1e-11)
    effects={key:math.fsum(v)/len(v) for key,v in ce.items()}
    expected_c=[dict(zip(KEYS+["component","half","effect"],key+(effect,))) for key,effect in sorted(effects.items())]
    pd.testing.assert_frame_equal(pd.DataFrame(expected_c),components,check_dtype=False,atol=1e-10,rtol=1e-11)
    for row in table.itertuples():
        key=(row.season,row.round,row.driver);g=saved[saved.season.eq(row.season)&saved["round"].eq(row.round)&saved.driver.eq(row.driver)]
        all_values=[v for k,v in effects.items() if k[:3]==key]
        full=math.fsum(all_values)/len(all_values) if all_values else np.nan
        np.testing.assert_allclose(row.full,full,atol=1e-10,rtol=1e-11,equal_nan=True)
        assert row.pairs==len(g) and row.components==len(all_values) and row.original_supported==bool(support_map[key].supported)
        ok=row.original_supported
        for h in (0,1):
            selected=g[g.half.eq(h)];vals=[v for k,v in effects.items() if k[:3]==key and k[4]==h]
            half=math.fsum(vals)/len(vals) if vals else np.nan
            np.testing.assert_allclose(getattr(row,f"half{h}"),half,atol=1e-10,rtol=1e-11,equal_nan=True)
            n=len(selected);on=selected.on_lap.nunique();off=selected.off_lap.nunique();nunits=len(selected[["on_lap","off_lap"]].drop_duplicates())
            assert (getattr(row,f"pairs{h}"),getattr(row,f"on_laps{h}"),getattr(row,f"off_laps{h}"),getattr(row,f"lap_pairs{h}"))==(n,on,off,nunits)
            ok=ok and n>=5 and on>=2 and off>=2 and nunits>=2
        assert row.repeat_supported==ok
    qualified=0
    for p in result["pilots"]:
        g=table[table["round"].eq(p["round"])&table.repeat_supported].sort_values("driver")
        r=pearson(g.half0.tolist(),g.half1.tolist())
        assert len(g)==p["supported"] and p["qualified"]==(len(g)>=6)
        np.testing.assert_allclose(r,p["correlation"],atol=1e-10,rtol=1e-11)
        qualified+=p["qualified"]
    assert qualified<2 and not result["coverage_gate"] and not result["gate"]
    assert result["reason"]=="insufficient_lap_disjoint_support" and result["bootstrap_finite"]==result["identity_null_finite"]==0
    assert all(result[k] is None for k in ("pooled_correlation","bootstrap_ci","identity_null_tail"))
    assert pd.read_csv(ROOT/"results/e263_scored.csv").empty
    assert not result["winner_fit"] and not result["temporal_forecast"] and not result["physical_causality_verified"]
    for name,digest in result["files"].items():assert hashlib.sha256((ROOT/"results"/name).read_bytes()).hexdigest()==digest
    print("E263 all1626SVD/unionfind/disjoint-lap-and-frame/weights/all60rows/support/correlation/no-resampling/hash/gate PASS")


if __name__=="__main__":main()
