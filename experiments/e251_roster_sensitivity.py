"""E251 within-season previous-roster sensitivity; not announcement-time proof."""
import json

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.e168_winner import win_score
from f1.data import ROOT,DATA,load_results
from f1.features import build_features,pre_weekend_cols

KEY=["season","round"]
MODELS=("standings","pre_logit","ens","win_logit","win_null")


def prior_roster(results,season,round_):
    past=results[results.season.eq(season)&results["round"].lt(round_)]
    if past.empty:return None
    last=past[past["round"].eq(past["round"].max())][["driver","code","team"]].sort_values("driver").reset_index(drop=True)
    assert not last.driver.duplicated().any() and not last.code.duplicated().any()
    return last


def history_target(results,pace,season,round_):
    assert results.season.le(2022).all() and pace.season.le(2022).all()
    roster=prior_roster(results,season,round_)
    if roster is None:raise ValueError("first-round roster is not supplied by history")
    current=results[results.season.eq(season)&results["round"].eq(round_)]
    metadata=current[["season","round","date","race","circuit"]].drop_duplicates()
    assert len(metadata)==1
    new=roster.assign(**metadata.iloc[0].to_dict())
    for c in results.columns:
        if c not in new:new[c]=np.nan
    new["status"]="Unknown";new["grid"]=0
    before=(results.season.lt(season)|(results.season.eq(season)&results["round"].lt(round_)))
    raw=pd.concat([results[before],new[results.columns]],ignore_index=True)
    x=build_features(raw,pace)
    return x[x.season.eq(season)&x["round"].eq(round_)].copy()


def score(train,current,F):
    assert train.k.max()<current.k.min()
    W.assert_pre_weekend(F)
    out={"standings":(-current.d_season_pts.fillna(0)*100+current.d10_order.fillna(11)*1e-3).to_numpy(),
         "pre_logit":W.m_logit_pod(train,current,F,C=.03,hl=100),"win_logit":win_score(train,current,F),
         "win_null":win_score(train,current,F,shuffled=True,seed=168)}
    out["ens"]=np.mean([W._rank01(out[n]) for n in ("standings","pre_logit")],axis=0)
    return out


def evaluate(predictions,truth):
    assert not predictions.duplicated(KEY+["driver","model","policy"]).any()
    rows=[]
    for (season,round_,policy,model),g in predictions.groupby(KEY+["policy","model"],sort=True):
        target=truth[truth.season.eq(season)&truth["round"].eq(round_)].sort_values("order")
        p=g.sort_values(["score","driver"])
        rows.append(dict(season=int(season),round=int(round_),policy=policy,model=model,pick=p.driver.iloc[0],
                         winner=int(p.driver.iloc[0]==target.driver.iloc[0]),
                         podium=len(set(p.driver.iloc[:3])&set(target.driver.iloc[:3]))/3))
    return pd.DataFrame(rows)


def main():
    results=load_results().loc[lambda d:d.season.le(2022)].copy()
    pace=pd.read_csv(DATA/"racepace_ff1.csv").loc[lambda d:d.season.le(2022)]
    X=build_features(results,pace);F=pre_weekend_cols(X);W.assert_pre_weekend(F)
    old=pd.read_csv(ROOT/"results/e214_inner_predictions.csv",float_precision="round_trip")
    parts=[];events=[]
    for (season,round_),current in X[X.season.between(2020,2022)].groupby(KEY,sort=True):
        train=X[X.k.lt(current.k.iloc[0])&X.season.ge(2019)]
        actual=score(train,current,F)
        for name in ("standings","pre_logit","ens","win_logit"):
            reference=old[old.season.eq(season)&old["round"].eq(round_)&old.model.eq(name)].set_index("driver").score
            np.testing.assert_allclose(actual[name],reference.reindex(current.driver),atol=1e-14,rtol=0)
        prior=prior_roster(results,season,round_)
        now=set(map(tuple,current[["driver","code","team"]].to_numpy()))
        previous=set(map(tuple,prior[["driver","code","team"]].to_numpy())) if prior is not None else now
        changed=now!=previous
        event=dict(season=int(season),round=int(round_),first_round_assumed=prior is None,changed=changed,
                   actual_only=[list(p) for p in sorted(now-previous)],prior_only=[list(p) for p in sorted(previous-now)])
        if changed:
            alternate=history_target(results,pace,season,round_)
            assert list(pre_weekend_cols(alternate))==F
            alternative=score(train,alternate,F)
        else:alternate=current;alternative=actual
        for policy,cur,scores in (("actual_roster",current,actual),("previous_roster",alternate,alternative)):
            for name,values in scores.items():
                parts.append(cur[KEY+["driver"]].assign(policy=policy,model=name,score=values))
        events.append(event)
    P=pd.concat(parts,ignore_index=True);R=evaluate(P,results)
    assert R.groupby(["policy","model"]).size().eq(61).all()
    comparison={}
    for model in MODELS:
        a=R[R.policy.eq("actual_roster")&R.model.eq(model)].set_index(KEY).sort_index()
        b=R[R.policy.eq("previous_roster")&R.model.eq(model)].set_index(KEY).sort_index()
        delta,ci,p=W.paired_boot(b.winner,a.winner)
        comparison[model]=dict(actual_hits=int(a.winner.sum()),previous_hits=int(b.winner.sum()),
            changed_picks=int(a.pick.ne(b.pick).sum()),winner_difference=float(delta),ci=ci.tolist(),p_nonpositive=float(p),
            actual_podium=float(a.podium.mean()),previous_podium=float(b.podium.mean()))
    summary=dict(races=61,within_season_roster_changes=sum(e["changed"] for e in events),events=events,
                 comparisons=comparison,announcement_times_verified=False,first_round_rosters_certified=False,
                 leakage_proven=False,production_changed=False)
    P.to_csv(ROOT/"results/e251_predictions.csv",index=False);R.to_csv(ROOT/"results/e251_per_race.csv",index=False)
    (ROOT/"results/e251_roster_sensitivity.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
