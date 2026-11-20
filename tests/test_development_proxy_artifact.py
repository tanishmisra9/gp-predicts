"""Independent daily integration, scalar aggregation and expanded bootstrap audit."""
import hashlib
import json
import math
from datetime import date,timedelta

import numpy as np
import pandas as pd

from experiments.b6_adaptation_gate import LINEAGE
from f1.data import ROOT,load_results


def main():
    artifact=json.loads((ROOT/"results/e243_development_proxy.json").read_text())
    sourcepath=ROOT/"results/e242_constructor_snapshots.json"
    errorpath=ROOT/"results/e194_forecasts_errors_scales.csv"
    assert artifact["source_sha256"]==hashlib.sha256(sourcepath.read_bytes()).hexdigest()
    assert artifact["errors_sha256"]==hashlib.sha256(errorpath.read_bytes()).hexdigest()
    sources=json.loads(sourcepath.read_text())["sources"]
    forecasts=pd.read_csv(errorpath,float_precision="round_trip")
    assert forecasts.season.max()==2022
    history=forecasts[forecasts.season.between(2021,2022)]
    calendar=load_results().loc[lambda z:z.season.between(2021,2022)]
    days={(int(r.season),int(r.round)):r.date.date() for r in calendar.itertuples()}
    rows=artifact["analysis_rows"]; keys=set()
    for r in rows:
        key=(r["season"],r["round"],r["k"],r["team"])
        assert key not in keys; keys.add(key)
        day=days[key[:2]]
        while day.weekday()!=3:day-=timedelta(days=1)
        assert r["cutoff"]==day.isoformat()
        end=day-timedelta(days=180); start=max(date(2021,1,1),end-timedelta(days=365))
        total=0.; counts={}; latest=0.
        for j in range(max(0,(end-start).days)):
            current=start+timedelta(days=j)
            active=[s for s in sources if date.fromisoformat(s["reference_date"])<=current]
            assert active
            s=active[-1]
            assert date.fromisoformat(s["event_date"])<date.fromisoformat(s["reference_date"])<=current<end<day
            table={t["constructor"]:t["position"] for t in s["constructors"]}
            if r["team"] in table:rank=table[r["team"]]
            else:
                matches=[v for k,v in table.items() if LINEAGE.get(k,k)==LINEAGE.get(r["team"],r["team"])]
                assert len(matches)==1; rank=matches[0]
            latest=(rank-5.5)/9; total+=latest
            sk=(s["season"],s["round"],rank); counts[sk]=counts.get(sk,0)+1
        assert abs(r["proxy"]-total/365)<1e-12 and r["oldrank"]==latest
        assert counts=={(u["season"],u["round"],u["rank"]):u["days"] for u in r["used"]}
        g=history[(history.season==key[0])&(history["round"]==key[1])&(history.k==key[2])&(history.team==key[3])]
        assert len(g)>0
        for field in ("error","mean_forecast"):
            vals=[float(v) for v in g[field] if pd.notna(v)]
            assert (r[field] is None and not vals) or abs(r[field]-math.fsum(vals)/len(vals))<1e-12
    assert keys==set(map(tuple,history[["season","round","k","team"]].drop_duplicates().to_numpy()))
    d=pd.DataFrame(rows)
    for _,g in d.groupby("k",sort=True):
        g=g.dropna(subset=["error","mean_forecast"])
        if len(g)<7 or g.proxy.std()<1e-10:
            assert g.x_centered.isna().all() and g.x_partial.isna().all(); continue
        assert np.allclose(g.x_centered,g.proxy-g.proxy.mean(),atol=1e-12)
        assert np.allclose(g.y_centered,g.error-g.error.mean(),atol=1e-12)
        basis=[]
        for v in (np.ones(len(g)),g.oldrank.to_numpy(),g.mean_forecast.to_numpy()):
            v=v.copy()
            for b in basis:v-=b@v*b
            norm=np.linalg.norm(v)
            if norm>1e-10:basis.append(v/norm)
        def project(v):
            v=v.copy()
            for b in basis:v-=b@v*b
            return v
        x=project(g.proxy.to_numpy()); y=project(g.error.to_numpy())
        if np.std(x)<1e-10:assert g.x_partial.isna().all()
        else:
            assert np.allclose(x,g.x_partial,atol=1e-10)
            assert np.allclose(y,g.y_partial,atol=1e-10)
    def check(frame,x,y,expected):
        assert len(frame)==expected["rows"] and frame.k.nunique()==expected["races"]
        if frame.empty:assert expected["correlation"] is None; return
        groups=[g[[x,y]].to_numpy() for _,g in frame.groupby("k",sort=True)]
        actual=np.corrcoef(frame[x],frame[y])[0,1]
        assert abs(actual-expected["correlation"])<1e-12
        rng=np.random.default_rng(243); samples=[]
        for _ in range(4000):
            indices=rng.integers(len(groups),size=len(groups))
            z=np.concatenate([groups[i] for i in indices])
            samples.append(np.corrcoef(z.T)[0,1])
        assert np.allclose(np.percentile(samples,[2.5,97.5]),expected["ci"],atol=1e-11)
    primary=d.dropna(subset=["x_centered","y_centered"])
    partial=d.dropna(subset=["x_partial","y_partial"])
    check(primary,"x_centered","y_centered",artifact["primary"])
    check(partial,"x_partial","y_partial",artifact["partial"])
    rng=np.random.default_rng(243); shuffled=primary.copy()
    for _,ids in shuffled.groupby("k",sort=True).groups.items():
        shuffled.loc[ids,"y_centered"]=rng.permutation(shuffled.loc[ids,"y_centered"].to_numpy())
    check(shuffled,"x_centered","y_centered",artifact["shuffled"])
    for year,g in primary.groupby("season"):
        check(g,"x_centered","y_centered",artifact["by_season"][str(year)])
    assert artifact["partial_by_season"]=={str(y):g.k.nunique() for y,g in partial.groupby("season")}
    def passes(a,races,n):
        return a["races"]>=races and a["rows"]>=n and a["correlation"]<=-.15 and a["ci"][1]<0
    gate=passes(artifact["primary"],30,250) and passes(artifact["partial"],18,150) and artifact["primary"]["correlation"]<artifact["shuffled"]["correlation"]
    assert artifact["gate"]==gate
    assert not any(artifact[k] for k in ("winner_fit","historical_publication_verified","entitlement_verified"))
    print("independent daily integration/scalar means/orthogonal projections/expanded bootstrap/null/gate audit passed")


if __name__=="__main__":main()
