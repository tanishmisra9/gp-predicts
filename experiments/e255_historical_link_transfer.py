"""E255 fixed prequential Q1 graph transfer; measurement, not winner model."""
import hashlib
import json

import numpy as np
import pandas as pd

import f1.walkforward as W
from f1.data import ROOT,DATA,load_results

KEY=["season","round","team","a","b"]
MODELS=("historical","modern","direct","shuffled")


def pairs(results):
    assert results.season.between(2014,2022).all()
    assert not results.duplicated(["season","round","driver"]).any()
    rows=[]
    for (season,round_,team),g in results.groupby(["season","round","team"],sort=True):
        if len(g)!=2:continue
        g=g.sort_values("driver");a,b=g.iloc[0],g.iloc[1]
        assert a.date==b.date and a.driver!=b.driver
        q=g.q1.to_numpy(float)
        gap=float(100*np.log(q[0]/q[1])) if np.isfinite(q).all() and (q>0).all() else np.nan
        rows.append(dict(season=int(season),round=int(round_),team=team,a=a.driver,b=b.driver,date=a.date,gap=gap))
    return pd.DataFrame(rows,columns=KEY+["date","gap"])


def shuffled(source):
    out=source.sort_values(KEY).reset_index(drop=True).copy();rng=np.random.default_rng(255)
    for _,ids in out[out.gap.notna()].groupby(["season","round"],sort=True).groups.items():
        out.loc[ids,"gap"]=rng.permutation(out.loc[ids,"gap"].to_numpy())
    return out


def past_rows(source,current):
    assert source.season.between(2014,2022).all() and current.season.between(2020,2022).all()
    assert not source.duplicated(KEY).any() and current[["season","round","date"]].drop_duplicates().shape[0]==1
    season,round_=int(current.season.iloc[0]),int(current["round"].iloc[0])
    date=current.date.iloc[0];assert date.dayofweek==6
    cutoff=date-pd.Timedelta(days=3)
    before=source.season.lt(season)|(source.season.eq(season)&source["round"].lt(round_))
    past=source[before&source.date.lt(cutoff)&source.gap.notna()].copy()
    assert np.isfinite(past.gap).all()
    return past,cutoff


def eligible(past,include_old):
    edges=set()
    blocks=(past[past.season.ge(2018)],past[past.season.le(2017)]) if include_old else (past[past.season.ge(2018)],)
    for block in blocks:
        counts=block.groupby(["a","b"]).size()
        edges.update(pair for pair,n in counts.items() if n>=3)
    selected=pd.MultiIndex.from_frame(past[["a","b"]]).isin(edges)
    return past[selected&(include_old|past.season.ge(2018))].copy()


def fit(source,current,include_old):
    W.assert_pre_weekend(["prior_q1_graph_gap"])
    past,cutoff=past_rows(source,current);train=eligible(past,include_old)
    drivers=sorted(set(train.a)|set(train.b)|set(current.a)|set(current.b));index={d:i for i,d in enumerate(drivers)}
    design=np.zeros((len(train),len(drivers)))
    for i,r in enumerate(train.itertuples()):design[i,index[r.a]]=1;design[i,index[r.b]]=-1
    weights=.5**((cutoff-train.date).dt.total_seconds().to_numpy()/86400/730.5)
    rating=np.linalg.solve(design.T@(weights[:,None]*design)+np.eye(len(drivers)),design.T@(weights*train.gap.to_numpy()))
    prediction=np.array([rating[index[r.a]]-rating[index[r.b]] for r in current.itertuples()])
    state=dict(drivers=drivers,ratings=rating.tolist(),observations=len(train),cutoff=cutoff.isoformat())
    return prediction,state


def direct(source,current):
    past,cutoff=past_rows(source,current);past=past[past.season.ge(2018)]
    estimates=[];counts=[]
    for r in current.itertuples():
        g=past[past.a.eq(r.a)&past.b.eq(r.b)];counts.append(len(g))
        w=.5**((cutoff-g.date).dt.total_seconds().to_numpy()/86400/730.5)
        estimates.append(float(w@g.gap.to_numpy()/(1+w.sum())) if len(g)>=3 else 0.)
    return np.array(estimates),counts


def summarize(predictions):
    observed=predictions[predictions.gap.notna()].copy()
    losses={};accuracies={}
    for name in MODELS:
        observed[name+"_loss"]=(observed[name]-observed.gap)**2
        observed[name+"_accuracy"]=np.where(observed[name].eq(0)|observed.gap.eq(0),.5,(np.sign(observed[name])==np.sign(observed.gap)).astype(float))
        losses[name]=observed.groupby(["season","round"])[name+"_loss"].mean()
        accuracies[name]=observed.groupby(["season","round"])[name+"_accuracy"].mean()
    races=pd.DataFrame(losses);acc=pd.DataFrame(accuracies)
    compare={}
    for name in ("modern","direct","shuffled"):
        gain,ci,p=W.paired_boot(races[name],races.historical)
        compare[name]=dict(gain=float(gain),ci=ci.tolist(),p_nonpositive=float(p))
    reduction=compare["modern"]["gain"]/races.modern.mean()
    coverage=observed.groupby(["season","round"]).size()
    gate=bool(len(observed)>=500 and len(races)>=55 and coverage.ge(8).sum()>=50 and reduction>=.02 and
              all(v["gain"]>0 and v["ci"][0]>0 for v in compare.values()) and acc.historical.mean()>=acc.modern.mean())
    new=observed[observed.direct_count.lt(3)]
    result=dict(forecast_rows=len(predictions),forecast_races=int(predictions.groupby(["season","round"]).ngroups),
        labeled_rows=len(observed),labeled_races=len(races),races_with_eight_labels=int(coverage.ge(8).sum()),
        mse=races.mean().to_dict(),teammate_accuracy=acc.mean().to_dict(),paired=compare,relative_modern_mse_reduction=float(reduction),
        new_pair_rows=len(new),new_pair_mse={n:float(((new[n]-new.gap)**2).mean()) for n in MODELS},
        gate=gate,winner_evaluation=False,roster_cutoff_verified=False)
    return result,observed,races.reset_index(),acc.reset_index()


def main():
    old=pd.read_csv(DATA/"results_2014_2017.csv",parse_dates=["date"])
    recent=load_results().loc[lambda d:d.season.between(2018,2022)]
    source=pairs(pd.concat([old,recent],ignore_index=True));null=shuffled(source)
    parts=[];states=[]
    for (season,round_),current in source[source.season.between(2020,2022)].groupby(["season","round"],sort=True):
        assert len(current)==10
        out=current.copy()
        for name,data,include_old in (("historical",source,True),("modern",source,False),("shuffled",null,True)):
            out[name],state=fit(data,current,include_old)
            states.append(dict(season=int(season),round=int(round_),model=name,**state))
        out["direct"],out["direct_count"]=direct(source,current)
        parts.append(out)
    predictions=pd.concat(parts,ignore_index=True)
    assert len(predictions)==610 and predictions.groupby(["season","round"]).ngroups==61
    result,observed,losses,acc=summarize(predictions)
    result["hashes"]={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ("results_2014_2017.csv","results.csv")}
    source.to_csv(ROOT/"results/e255_q1_source.csv",index=False)
    for name,d in (("predictions",predictions),("scored",observed),("race_losses",losses),("race_accuracy",acc)):
        d.to_csv(ROOT/f"results/e255_{name}.csv",index=False)
    (ROOT/"results/e255_states.json").write_text(json.dumps(states,indent=2)+"\n")
    (ROOT/"results/e255_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
