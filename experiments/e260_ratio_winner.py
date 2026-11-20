"""E260 one anonymous earlier-team episode rate, full inner winner test."""
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
import f1.walkforward as W
from experiments.e168_winner import win_score
from experiments.e190_traffic_feasibility import cached_session
from experiments.e258_gear_ratio import windows,measure
from f1.data import ROOT,DATA,load_results
from f1.features import build_features,pre_weekend_cols

NEW=["prior_team_ratio_episode_rate"]
STATUS=ROOT/"results/e260_source_status.json"


def observations(reports,entries):
    keys=["season","round","code"]
    assert not entries.duplicated(keys).any() and entries.season.between(2018,2022).all()
    legal=set(map(tuple,entries[keys].to_numpy())); mapping={}
    assert len({(r["season"],r["round"]) for r in reports})==len(reports)
    for r in reports:
        assert 2018<=r["season"]<=2022
        for row in r.get("rows",[]):
            key=(r["season"],r["round"],row["driver"])
            assert key in legal and key not in mapping
            if row["available"]:
                assert 0<=row["episodes"]<=row["eligible"] and row["supported_gears"]>=0
                assert all(np.isfinite(row[c]) for c in ("episodes","eligible","supported_gears"))
            qualified=row["available"] and row["eligible"]>=100 and row["supported_gears"]>=2
            mapping[key]=(row["eligible"],row["episodes"]) if qualified else (np.nan,np.nan)
    out=entries[keys].copy().reset_index(drop=True)
    out[["eligible","episodes"]]=[mapping.get(tuple(k),(np.nan,np.nan)) for k in out.to_numpy()]
    return out


def add_history(X,table):
    keys=["season","round","code"]
    assert not table.duplicated(keys).any() and table.season.between(2018,2022).all()
    assert table.eligible.isna().equals(table.episodes.isna())
    valid=table.eligible.notna()
    assert table.loc[valid,"eligible"].ge(table.loc[valid,"episodes"]).all() and table.loc[valid,"episodes"].ge(0).all()
    assert np.isfinite(table.loc[valid,["eligible","episodes"]].to_numpy()).all()
    metadata=X[keys+["k","team","date"]]
    history=table.merge(metadata,on=keys,validate="one_to_one")
    assert len(history)==len(table)
    out=X.copy();out[NEW[0]]=.5
    for k,current in X.groupby("k",sort=True):
        cutoff=pd.Timestamp(current.date.iloc[0]).normalize()-pd.Timedelta(days=3)
        past=history[history.k.lt(k)&history.date.lt(cutoff)&history.eligible.notna()]
        w=.5**((k-past.k)/10)
        global_rate=(float((w*past.episodes).sum())+.5)/(float((w*past.eligible).sum())+1)
        numerator=(w*past.episodes).groupby(past.team).sum()
        denominator=(w*past.eligible).groupby(past.team).sum()
        rate=(numerator+10000*global_rate)/(denominator+10000)
        out.loc[current.index,NEW[0]]=current.team.map(rate).fillna(global_rate)
    return out


def extract():
    import fastf1
    import f1.ff1
    inventory=json.loads((ROOT/"results/e212_source_status.json").read_text())
    assert len(inventory)==103 and sum(r["available"] for r in inventory)==99
    reports=json.loads(STATUS.read_text()) if STATUS.exists() else []
    done={(r["season"],r["round"]) for r in reports}
    assert len(done)==len(reports) and done.issubset({(r["season"],r["round"]) for r in inventory})
    fastf1.Cache.offline_mode(True)
    for receipt in inventory:
        season,round_=receipt["season"],receipt["round"]
        if (season,round_) in done:continue
        report=dict(season=season,round=round_,available=False,rows=[],events=[])
        if not receipt["available"]:report["reason"]="unavailable_in_frozen_inventory"
        else:
            try:
                with patch("requests.adapters.HTTPAdapter.send",side_effect=RuntimeError("E260 HTTP forbidden")):
                    session=cached_session(fastf1,season,round_)
                    laps=pd.DataFrame(session.laps)
                    cars=fastf1._api.car_data(session.api_path)
            except Exception as exc:report["error_type"]=type(exc).__name__
            else:
                selected=windows(laps)
                origin=max((c.Date-c.Time).max() for c in cars.values() if c.Time.notna().any()).round("ms")
                for driver,raw in laps.groupby("Driver",sort=True):
                    numbers=raw.DriverNumber.dropna().unique();assert len(numbers)==1
                    car=cars.get(str(numbers[0]))
                    if car is None:report["rows"].append(dict(driver=driver,available=False))
                    else:
                        support,episodes=measure(car,selected[selected.Driver.eq(driver)],origin)
                        report["rows"].append(dict(driver=driver,available=True,**support))
                        report["events"].extend(episodes.assign(driver=driver).to_dict("records"))
                report["available"]=True
        reports.append(report);STATUS.write_text(json.dumps(reports,indent=2)+"\n")
        print("SOURCE",season,round_,report["available"],sum(r.get("episodes",0) for r in report["rows"]),flush=True)
    entries=load_results().loc[lambda d:d.season.between(2018,2022),["season","round","code"]]
    table=observations(reports,entries)
    coverage=table.assign(qualified=table.eligible.notna()).groupby("season").qualified.mean().to_dict()
    summary=dict(events=len(reports),available=sum(r["available"] for r in reports),coverage=coverage,
                 episodes=int(table.episodes.sum()),gate=bool(len(reports)==103 and sum(r["available"] for r in reports)>=95 and min(coverage.values())>=.8))
    table.to_csv(ROOT/"results/e260_observations.csv",index=False)
    (ROOT/"results/e260_source_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    assert summary["gate"],"registered source coverage failed"
    return table


def main():
    source=extract()
    df=load_results().loc[lambda d:d.season.le(2022)]
    pace=pd.read_csv(DATA/"racepace_ff1.csv").loc[lambda d:d.season.le(2022)]
    X=build_features(df,pace).merge(df[["season","round","code","date"]],on=["season","round","code"],validate="one_to_one");base=pre_weekend_cols(X);X=add_history(X,source)
    features=base+NEW;W.assert_pre_weekend(features)
    W.MODELS.update(standings=lambda tr,te,F:(-te.d_season_pts.fillna(0)*100+te.d10_order.fillna(11)*1e-3).to_numpy(),
                    pre_logit=lambda tr,te,F:W.m_logit_pod(tr,te,base,C=.03,hl=100),
                    win_logit=lambda tr,te,F:win_score(tr,te,base),ratio_win=lambda tr,te,F:win_score(tr,te,features),
                    ratio_null=lambda tr,te,F:win_score(tr,te,features,shuffled=True,seed=260))
    R,P=W.walk_forward(X,[2020,2021,2022],["standings","pre_logit","win_logit","ratio_win","ratio_null"],F=features,
                       ens={"ens":["standings","pre_logit"]},tie=None)
    assert len(R)==61*6 and len(P)==1220*6
    R.to_csv(ROOT/"results/e260_inner_per_race.csv",index=False)
    P.to_csv(ROOT/"results/e260_inner_predictions.csv",index=False)
    X[["season","round","code","k"]+NEW].to_csv(ROOT/"results/e260_features.csv",index=False)
    candidate=R[R.model.eq("ratio_win")].set_index(["season","round"]).winner
    paired={}
    for name in ("standings","ens","win_logit"):
        reference=R[R.model.eq(name)].set_index(["season","round"]).winner
        delta,ci,p=W.paired_boot(candidate,reference)
        paired[name]=dict(difference=float(delta),ci=ci.tolist(),p_nonpositive=float(p))
    hits=R.groupby("model").winner.sum().astype(int).to_dict()
    summary=dict(hits=hits,overlap=R.groupby("model").podium.mean().to_dict(),paired=paired,
                 gate=bool(all(v["ci"][0]>0 for v in paired.values()) and hits["ratio_win"]>hits["ratio_null"]))
    (ROOT/"results/e260_inner_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("RESULT",json.dumps(summary),flush=True)


if __name__=="__main__":main()
