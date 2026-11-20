"""E251 independent roster chronology, ranking, full denominator and bootstrap."""
import json

import numpy as np
import pandas as pd

from f1.data import ROOT,load_results


def main():
    raw=load_results().loc[lambda d:d.season.le(2022)]
    P=pd.read_csv(ROOT/"results/e251_predictions.csv",float_precision="round_trip")
    R=pd.read_csv(ROOT/"results/e251_per_race.csv",float_precision="round_trip")
    report=json.loads((ROOT/"results/e251_roster_sensitivity.json").read_text())
    old=pd.read_csv(ROOT/"results/e214_inner_predictions.csv",float_precision="round_trip")
    key=["season","round"]
    assert P.season.between(2020,2022).all() and not P.duplicated(key+["policy","model","driver"]).any()
    models=("standings","pre_logit","ens","win_logit","win_null")
    expected=[];events=[];last={}
    for (season,round_),g in raw.groupby(key,sort=True):
        now=set(map(tuple,g[["driver","code","team"]].to_numpy()))
        prior=last.get(season);last[season]=now
        if season<2020:continue
        assumed=prior is None
        if prior is None:prior=now
        events.append(dict(season=int(season),round=int(round_),first_round_assumed=assumed,changed=now!=prior,
                           actual_only=[list(p) for p in sorted(now-prior)],prior_only=[list(p) for p in sorted(prior-now)]))
        actual_ids={p[0] for p in now};previous_ids={p[0] for p in prior}
        target=g.sort_values("order")
        assert target.order.eq(1).sum()==1
        truth=target.driver.tolist()
        for policy,drivers in (("actual_roster",actual_ids),("previous_roster",previous_ids)):
            for model in models:
                part=P[P.season.eq(season)&P["round"].eq(round_)&P.policy.eq(policy)&P.model.eq(model)]
                assert set(part.driver)==drivers and len(part)==len(drivers)
                pairs=sorted(zip(part.score,part.driver))
                picked=[d for s,d in pairs]
                expected.append(dict(season=int(season),round=int(round_),policy=policy,model=model,pick=picked[0],
                    winner=int(picked[0]==truth[0]),podium=sum(d in set(truth[:3]) for d in picked[:3])/3))
                if policy=="actual_roster" and model!="win_null":
                    ref=old[old.season.eq(season)&old["round"].eq(round_)&old.model.eq(model)].set_index("driver").score
                    np.testing.assert_allclose(part.score,ref.reindex(part.driver),atol=1e-14,rtol=0)
    expected=pd.DataFrame(expected).sort_values(key+["policy","model"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(R.sort_values(key+["policy","model"]).reset_index(drop=True),expected,check_dtype=False,atol=1e-15)
    assert report["events"]==events and report["within_season_roster_changes"]==sum(e["changed"] for e in events)
    assert len(events)==61 and report["races"]==61 and sum(e["first_round_assumed"] for e in events)==3
    assert R.groupby(["policy","model"]).size().eq(61).all()
    for model in models:
        a=expected[expected.policy.eq("actual_roster")&expected.model.eq(model)].set_index(key).sort_index()
        b=expected[expected.policy.eq("previous_roster")&expected.model.eq(model)].set_index(key).sort_index()
        delta=(b.winner-a.winner).to_numpy(float)
        indices=np.random.default_rng(0).integers(61,size=(5000,61))
        draws=delta[indices].sum(axis=1)/61
        s=report["comparisons"][model]
        assert (int(a.winner.sum()),int(b.winner.sum()),int(a.pick.ne(b.pick).sum()))==(s["actual_hits"],s["previous_hits"],s["changed_picks"])
        np.testing.assert_allclose(s["winner_difference"],sum(delta)/61,atol=1e-15)
        np.testing.assert_allclose(s["ci"],np.quantile(draws,[.025,.975]),atol=1e-15)
        assert s["p_nonpositive"]==(draws<=0).mean()
        np.testing.assert_allclose([s["actual_podium"],s["previous_podium"]],[a.podium.mean(),b.podium.mean()],atol=1e-15)
    assert not any(report[k] for k in ("announcement_times_verified","first_round_rosters_certified","leakage_proven","production_changed"))
    print("E251 independent chronology/actual-vs-prior identities/full61 scores/winner+podium/ref/paired bootstrap/limited-claim audit PASS")


if __name__=="__main__":main()
