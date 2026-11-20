"""E255 scalar source/edge/direct reconstruction and augmented-SVD audit."""
import csv
import hashlib
import json
import math
from collections import defaultdict,Counter
from statistics import mean

import numpy as np
import pandas as pd

from f1.data import ROOT,DATA
from experiments.e255_historical_link_transfer import fit,direct,shuffled

KEY=["season","round","team","a","b"]


def main():
    report=json.loads((ROOT/"results/e255_summary.json").read_text())
    P=pd.read_csv(ROOT/"results/e255_predictions.csv",parse_dates=["date"],float_precision="round_trip")
    source=pd.read_csv(ROOT/"results/e255_q1_source.csv",parse_dates=["date"],float_precision="round_trip")
    states=json.loads((ROOT/"results/e255_states.json").read_text())
    state={(r["season"],r["round"],r["model"]):r for r in states}
    groups=defaultdict(list);seen=set()
    for name,years in (("results_2014_2017.csv",set(range(2014,2018))),("results.csv",set(range(2018,2023)))):
        path=DATA/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==report["hashes"][name]
        with path.open() as stream:
            for r in csv.DictReader(stream):
                y=int(r["season"])
                if y not in years:continue
                rd=int(r["round"]);driver=r["driver"]
                assert (y,rd,driver) not in seen;seen.add((y,rd,driver))
                groups[(y,rd,r["team"])].append((driver,pd.Timestamp(r["date"]),float(r["q1"] or "nan")))
    raw=[]
    for (y,rd,t),rows in sorted(groups.items()):
        if len(rows)!=2:continue
        (a,date,qa),(b,other,qb)=sorted(rows)
        assert date==other
        gap=100*math.log(qa/qb) if math.isfinite(qa) and math.isfinite(qb) and qa>0 and qb>0 else np.nan
        raw.append(dict(season=y,round=rd,team=t,a=a,b=b,date=date,gap=gap))
    expected=pd.DataFrame(raw,columns=KEY+["date","gap"])
    pd.testing.assert_frame_equal(source,expected,check_dtype=False,atol=1e-11,rtol=0)
    null=expected.sort_values(KEY).reset_index(drop=True).copy();rng=np.random.default_rng(255)
    for _,ids in null[null.gap.notna()].groupby(["season","round"],sort=True).groups.items():
        null.loc[ids,"gap"]=rng.permutation(null.loc[ids,"gap"].to_numpy())
    target=expected[expected.season.ge(2020)]
    assert len(target)==610 and target.groupby(["season","round"]).ngroups==61
    pd.testing.assert_frame_equal(P[KEY+["date","gap"]],target.reset_index(drop=True),check_dtype=False,atol=1e-11,rtol=0)
    for (y,rd),g in P.groupby(["season","round"],sort=True):
        cutoff=g.date.iloc[0]-pd.Timedelta(days=3)
        def previous(frame):
            return [r for r in frame.itertuples() if (r.season,r.round)<(y,rd) and r.date<cutoff and pd.notna(r.gap)]
        rawpast=previous(expected)
        for model,data,old in (("historical",expected,True),("modern",expected,False),("shuffled",null,True)):
            past=previous(data)
            counts={}
            for oldblock in ((False,True) if old else (False,)):
                appearances=defaultdict(set)
                for r in past:
                    if (r.season<2018)==oldblock:appearances[(r.a,r.b)].add((r.season,r.round))
                counts.update({p:len(events) for p,events in appearances.items() if len(events)>=3})
            train=[r for r in past if (old or r.season>=2018) and (r.a,r.b) in counts]
            drivers=sorted(set(d for r in train for d in (r.a,r.b))|set(g.a)|set(g.b))
            index={d:i for i,d in enumerate(drivers)}
            weighted=[];labels=[]
            for r in train:
                weight=math.exp(math.log(.5)*(cutoff-r.date).total_seconds()/86400/730.5)
                row=np.zeros(len(drivers));row[index[r.a]]=math.sqrt(weight);row[index[r.b]]=-math.sqrt(weight)
                weighted.append(row);labels.append(math.sqrt(weight)*r.gap)
            matrix=np.vstack([np.asarray(weighted).reshape(-1,len(drivers)),np.eye(len(drivers))])
            rhs=np.r_[labels,np.zeros(len(drivers))]
            rating=np.linalg.lstsq(matrix,rhs,rcond=None)[0]
            s=state[(y,rd,model)]
            assert s["drivers"]==drivers and s["observations"]==len(train) and s["cutoff"]==cutoff.isoformat()
            np.testing.assert_allclose(s["ratings"],rating,atol=1e-10,rtol=0)
            predicted=[rating[index[r.a]]-rating[index[r.b]] for r in g.itertuples()]
            np.testing.assert_allclose(g[model],predicted,atol=1e-10,rtol=0)
        for r in g.itertuples():
            past=[p for p in rawpast if p.season>=2018 and (p.a,p.b)==(r.a,r.b)]
            weights=[math.exp(math.log(.5)*(cutoff-p.date).total_seconds()/86400/730.5) for p in past]
            value=math.fsum(w*p.gap for w,p in zip(weights,past))/(1+math.fsum(weights)) if len(past)>=3 else 0.
            assert r.direct_count==len(past) and abs(r.direct-value)<1e-10
    valid=P[P.gap.notna()].copy();names=("historical","modern","direct","shuffled")
    for name in names:
        valid[name+"_loss"]=(valid[name]-valid.gap)**2
        valid[name+"_accuracy"]=np.where((valid[name]==0)|(valid.gap==0),.5,(np.sign(valid[name])==np.sign(valid.gap)).astype(float))
    saved=pd.read_csv(ROOT/"results/e255_scored.csv",parse_dates=["date"],float_precision="round_trip")
    pd.testing.assert_frame_equal(valid.reset_index(drop=True),saved,check_dtype=False,atol=1e-12,rtol=0)
    losses=[];accuracies=[]
    for (y,rd),g in valid.groupby(["season","round"],sort=True):
        losses.append(dict(season=y,round=rd,**{n:mean(g[n+"_loss"]) for n in names}))
        accuracies.append(dict(season=y,round=rd,**{n:mean(g[n+"_accuracy"]) for n in names}))
    losses=pd.DataFrame(losses);accuracies=pd.DataFrame(accuracies)
    for kind,frame in (("race_losses",losses),("race_accuracy",accuracies)):
        saved=pd.read_csv(ROOT/f"results/e255_{kind}.csv",float_precision="round_trip")
        pd.testing.assert_frame_equal(frame,saved,check_dtype=False,atol=1e-12,rtol=0)
    for name in names:
        assert abs(mean(losses[name])-report["mse"][name])<1e-12
        assert abs(mean(accuracies[name])-report["teammate_accuracy"][name])<1e-12
    for name in ("modern","direct","shuffled"):
        d=(losses[name]-losses.historical).to_numpy()
        draws=d[np.random.default_rng(0).integers(len(d),size=(5000,len(d)))].mean(axis=1)
        result=report["paired"][name]
        assert abs(mean(d)-result["gain"])<1e-12
        np.testing.assert_allclose(np.quantile(draws,[.025,.975]),result["ci"],atol=1e-12,rtol=0)
        assert (draws<=0).mean()==result["p_nonpositive"]
    coverage=valid.groupby(["season","round"]).size();reduction=report["paired"]["modern"]["gain"]/mean(losses.modern)
    assert [len(P),P.groupby(["season","round"]).ngroups,len(valid),len(losses),int(coverage.ge(8).sum())]==[report[k] for k in ("forecast_rows","forecast_races","labeled_rows","labeled_races","races_with_eight_labels")]
    assert abs(reduction-report["relative_modern_mse_reduction"])<1e-12
    new=valid[valid.direct_count.lt(3)]
    assert len(new)==report["new_pair_rows"]
    for name in names:assert abs(float(((new[name]-new.gap)**2).mean())-report["new_pair_mse"][name])<1e-12
    gate=len(valid)>=500 and len(losses)>=55 and coverage.ge(8).sum()>=50 and reduction>=.02 and all(r["gain"]>0 and r["ci"][0]>0 for r in report["paired"].values()) and mean(accuracies.historical)>=mean(accuracies.modern)
    assert report["gate"]==gate and not report["winner_evaluation"] and not report["roster_cutoff_verified"]
    # Real-source poisoning/prefix checks call the new forecaster separately.
    current=source[source.season.eq(2022)&source["round"].eq(10)].copy()
    baseline={old:fit(source,current,old)[0] for old in (False,True)}
    changed=source.copy();future=changed.season.eq(2022)&changed["round"].ge(10)
    changed.loc[future,"gap"]=999.;changed.loc[future,"a"]="FUT_A";changed.loc[future,"b"]="FUT_B"
    target2=current.copy();target2["gap"]=-999
    for old in (False,True):np.testing.assert_array_equal(baseline[old],fit(changed,target2,old)[0])
    np.testing.assert_array_equal(fit(shuffled(source),current,True)[0],fit(shuffled(changed),target2,True)[0])
    prefix=source[(source.season<2022)|(source.season.eq(2022)&source["round"].lt(10))]
    np.testing.assert_array_equal(baseline[True],fit(prefix,current,True)[0])
    changed=source.copy();ids=changed.index[changed.season.eq(2017)&changed.a.eq("bottas")&changed.b.eq("hamilton")&changed.gap.notna()]
    assert len(ids)>0
    changed.loc[ids[0],"gap"]+=10
    assert np.any(fit(changed,current,True)[0]!=baseline[True])
    np.testing.assert_array_equal(fit(changed,current,False)[0],baseline[False])
    print("E255 independent raw log-gaps/chronology/edge qualification/SVD all183states+610forecasts/direct/null/metrics/bootstrap/gate/hash/real source mutations PASS")


if __name__=="__main__":main()
