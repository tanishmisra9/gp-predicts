"""E246 wet-compound matched teammate support only, no pace response."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.e190_traffic_feasibility import cached_session
from f1.data import ROOT,load_results

COLS=["team","a","b","lap","compound","age_a","age_b"]


def pairs(laps,entries):
    assert not laps.duplicated(["Driver","LapNumber"]).any()
    assert not entries.code.duplicated().any()
    line=laps[laps.Time.notna()&laps.Position.notna()].sort_values(["LapNumber","Position","Driver"],kind="stable").copy()
    line["ahead"]=line.groupby("LapNumber").Time.shift()
    line["gap"]=(line.Time-line.ahead).dt.total_seconds()
    d=line[line.LapNumber.gt(1)&line.LapTime.notna()&line.PitInTime.isna()&line.PitOutTime.isna()&
           line.TrackStatus.astype(str).eq("1")&line.IsAccurate.fillna(False)&
           line.Compound.isin(["INTERMEDIATE","WET"])&line.TyreLife.notna()&
           (line.Position.eq(1)|line.gap.gt(1.5))].copy()
    d["seconds"]=d.LapTime.dt.total_seconds()
    d=d[np.isfinite(d.seconds)&d.seconds.gt(0)&np.isfinite(d.TyreLife)&d.TyreLife.ge(0)]
    d=d[d.seconds.lt(1.07*d.groupby(["Driver","Compound"]).seconds.transform("median"))]
    assert set(d.Driver).issubset(set(entries.code))
    rows=[]
    for team,g in entries.groupby("team",sort=True):
        codes=sorted(g.code)
        if len(codes)!=2:continue
        a=d[d.Driver.eq(codes[0])];b=d[d.Driver.eq(codes[1])]
        match=a.merge(b,on=["LapNumber","Compound"],suffixes=("_a","_b"),validate="one_to_one")
        match=match[(match.TyreLife_a-match.TyreLife_b).abs().le(3)]
        for r in match.itertuples():
            rows.append(dict(team=team,a=codes[0],b=codes[1],lap=int(r.LapNumber),compound=r.Compound,
                             age_a=float(r.TyreLife_a),age_b=float(r.TyreLife_b)))
    return pd.DataFrame(rows,columns=COLS)


def support(source):
    assert not source.duplicated(["season","round","team","lap"]).any()
    rows=[]
    for (season,round_,team,a,b),g in source.groupby(["season","round","team","a","b"],sort=True):
        even=int(g.lap.mod(2).eq(0).sum());odd=len(g)-even
        rows.append(dict(season=int(season),round=int(round_),team=team,a=a,b=b,n=len(g),half0=even,half1=odd,
                         supported=bool(len(g)>=10 and even>=5 and odd>=5)))
    table=pd.DataFrame(rows,columns=["season","round","team","a","b","n","half0","half1","supported"])
    good=table[table.supported.eq(True)]
    drivers={}
    for r in good.itertuples():
        for d in (r.a,r.b):drivers[d]=drivers.get(d,0)+1
    races=good[["season","round"]].drop_duplicates()
    repeated=sum(n>=3 for n in drivers.values())
    report=dict(matched_laps=len(source),pair_events=len(table),supported_pairs=len(good),supported_races=len(races),
                supported_seasons=int(good.season.nunique()),drivers_three_wet_events=repeated,
                gate=bool(len(good)>=40 and len(races)>=8 and good.season.nunique()>=3 and repeated>=8),
                response_fit=False,winner_fit=False)
    return report,table


def main():
    import fastf1
    import f1.ff1
    entries=load_results().loc[lambda d:d.season.between(2018,2022),["season","round","code","team"]]
    events=entries[["season","round"]].drop_duplicates()
    assert len(events)==103
    fastf1.Cache.offline_mode(True);parts=[];reports=[]
    for season,round_ in events.itertuples(index=False,name=None):
        r=dict(season=int(season),round=int(round_),available=False)
        try:
            with patch("requests.adapters.HTTPAdapter.send",side_effect=RuntimeError("wet support HTTP forbidden")):
                session=cached_session(fastf1,int(season),int(round_));laps=pd.DataFrame(session.laps)
        except Exception as exc:r["error_type"]=type(exc).__name__
        else:
            current=pairs(laps,entries[entries.season.eq(season)&entries["round"].eq(round_)])
            parts.append(current.assign(season=season,round=round_))
            r.update(available=True,raw_laps=len(laps),matched_laps=len(current))
        reports.append(r);print("SOURCE",json.dumps(r),flush=True)
    source=pd.concat(parts,ignore_index=True)
    result,table=support(source)
    result.update(sessions=reports)
    source.to_csv(ROOT/"results/e246_matched_wet_laps.csv",index=False)
    table.to_csv(ROOT/"results/e246_pair_support.csv",index=False)
    (ROOT/"results/e246_support_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print("SUPPORT",json.dumps({k:v for k,v in result.items() if k!="sessions"}),flush=True)


if __name__=="__main__":main()
