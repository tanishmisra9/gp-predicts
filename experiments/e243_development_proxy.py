"""E243: fixed lagged rank proxy versus frozen pace errors; diagnostic only."""
import hashlib
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from experiments.b6_adaptation_gate import LINEAGE
from f1.data import ROOT, load_results

SOURCES=((2020,17,"2021-01-01"),(2021,8,"2021-07-01"),(2021,22,"2022-01-01"))


def exposure(sources,team,cutoff):
    end=cutoff-timedelta(days=180)
    start=max(date(2021,1,1),end-timedelta(days=365))
    used=[]; total=0.
    for i,s in enumerate(sources):
        ref=date.fromisoformat(s["reference_date"])
        stop=date.fromisoformat(sources[i+1]["reference_date"]) if i+1<len(sources) else end
        days=(min(stop,end)-max(ref,start)).days
        if days<=0: continue
        assert date.fromisoformat(s["event_date"])<ref<=end<cutoff
        exact=[r for r in s["constructors"] if r["constructor"]==team]
        matches=exact or [r for r in s["constructors"] if LINEAGE.get(r["constructor"],r["constructor"])==LINEAGE.get(team,team)]
        assert len(matches)==1,"ambiguous or missing constructor"
        rank=matches[0]["position"]
        assert type(rank) is int and 1<=rank<=10
        value=(rank-5.5)/9
        total+=days*value
        used.append(dict(season=s["season"],round=s["round"],days=days,rank=rank))
    return total/365, (used[-1]["rank"]-5.5)/9 if used else 0., used


def residuals(x,controls):
    return x-controls@np.linalg.lstsq(controls,x,rcond=None)[0]


def association(frame,x,y):
    groups=[]
    for _,g in frame.groupby("k",sort=True):
        a=g[x].to_numpy(); b=g[y].to_numpy()
        groups.append([len(g),a.sum(),b.sum(),a@a,b@b,a@b])
    if not groups:return dict(rows=0,races=0,correlation=None,ci=[None,None])
    stats=np.asarray(groups)
    def corr(v):
        n,a,b,aa,bb,ab=v.T
        den=np.sqrt(np.maximum(0,(aa-a*a/n)*(bb-b*b/n)))
        return np.divide(ab-a*b/n,den,out=np.full_like(den,np.nan),where=den>1e-12)
    estimate=float(corr(stats.sum(axis=0)[None,:])[0])
    rng=np.random.default_rng(243)
    draws=stats[rng.integers(len(stats),size=(4000,len(stats)))].sum(axis=1)
    sample=corr(draws); sample=sample[np.isfinite(sample)]
    return dict(rows=len(frame),races=len(groups),correlation=estimate if np.isfinite(estimate) else None,
                ci=np.percentile(sample,[2.5,97.5]).tolist() if len(sample) else [None,None])


def main():
    sourcepath=ROOT/"results/e242_constructor_snapshots.json"
    sources=json.loads(sourcepath.read_text())["sources"]
    assert [(s["season"],s["round"],s["reference_date"]) for s in sources]==list(SOURCES)
    assert all(s["qualified"] for s in sources)
    path=ROOT/"results/e194_forecasts_errors_scales.csv"
    d=pd.read_csv(path,float_precision="round_trip")
    assert d.season.max()==2022
    d=d[d.season.between(2021,2022)]
    calendar=load_results().loc[lambda z:z.season.between(2021,2022),["season","round","date"]].drop_duplicates()
    assert not calendar.duplicated(["season","round"]).any()
    team=d.groupby(["season","round","k","team"],sort=True).agg(error=("error","mean"),mean_forecast=("mean_forecast","mean")).reset_index()
    team=team.merge(calendar,on=["season","round"],validate="many_to_one")
    rows=[]
    for r in team.itertuples():
        day=r.date.date(); cutoff=day-timedelta(days=(day.weekday()-3)%7)
        x,oldrank,used=exposure(sources,r.team,cutoff)
        rows.append(dict(season=int(r.season),round=int(r.round),k=int(r.k),team=r.team,
                         cutoff=cutoff.isoformat(),proxy=x,oldrank=oldrank,used=used,
                         error=r.error,mean_forecast=r.mean_forecast))
    out=pd.DataFrame(rows); out["x_partial"]=np.nan; out["y_partial"]=np.nan
    out["x_centered"]=np.nan; out["y_centered"]=np.nan
    for _,g in out.groupby("k",sort=True):
        g=g.dropna(subset=["error","mean_forecast"])
        if len(g)<7 or g.proxy.std()<1e-10:continue
        out.loc[g.index,"x_centered"]=g.proxy-g.proxy.mean()
        out.loc[g.index,"y_centered"]=g.error-g.error.mean()
        controls=np.column_stack([np.ones(len(g)),g.oldrank,g.mean_forecast])
        x=residuals(g.proxy.to_numpy(),controls)
        if np.std(x)<1e-10:continue
        out.loc[g.index,"x_partial"]=x
        out.loc[g.index,"y_partial"]=residuals(g.error.to_numpy(),controls)
    primary=out.dropna(subset=["x_centered","y_centered"])
    partial=out.dropna(subset=["x_partial","y_partial"])
    full=association(primary,"x_centered","y_centered")
    sensitivity=association(partial,"x_partial","y_partial")
    shuffled=primary.copy(); rng=np.random.default_rng(243)
    for _,ids in shuffled.groupby("k",sort=True).groups.items():
        shuffled.loc[ids,"y_centered"]=rng.permutation(shuffled.loc[ids,"y_centered"].to_numpy())
    null=association(shuffled,"x_centered","y_centered")
    def passes(a,n,rows):
        return bool(a["races"]>=n and a["rows"]>=rows and a["correlation"] is not None and
                    a["correlation"]<=-.15 and a["ci"][1] is not None and a["ci"][1]<0)
    report=dict(primary=full,partial=sensitivity,shuffled=null,
                by_season={str(y):association(g,"x_centered","y_centered") for y,g in primary.groupby("season")},
                partial_by_season={str(y):int(g.k.nunique()) for y,g in partial.groupby("season")},
                gate=passes(full,30,250) and passes(sensitivity,18,150) and
                     null["correlation"] is not None and full["correlation"]<null["correlation"],
                source_sha256=hashlib.sha256(sourcepath.read_bytes()).hexdigest(),
                errors_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                winner_fit=False,historical_publication_verified=False,entitlement_verified=False,
                analysis_rows=out.replace({np.nan:None}).to_dict("records"))
    (ROOT/"results/e243_development_proxy.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("rows","analysis_rows")},indent=2))


if __name__=="__main__":main()
