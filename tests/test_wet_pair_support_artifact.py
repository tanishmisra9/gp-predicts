"""E246 independent scalar raw-cache eligibility, pairing and count audit."""
import json
import math
from collections import defaultdict,Counter
from statistics import median
from unittest.mock import patch

import pandas as pd

from experiments.e190_traffic_feasibility import cached_session
from f1.data import ROOT,load_results


def main():
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True)
    saved=pd.read_csv(ROOT/"results/e246_matched_wet_laps.csv",float_precision="round_trip")
    table=pd.read_csv(ROOT/"results/e246_pair_support.csv",float_precision="round_trip")
    report=json.loads((ROOT/"results/e246_support_summary.json").read_text())
    entries=load_results().loc[lambda d:d.season.between(2018,2022),["season","round","code","team"]]
    source=[]
    for receipt in report["sessions"]:
        y,r=receipt["season"],receipt["round"]
        try:
            with patch("requests.adapters.HTTPAdapter.send",side_effect=RuntimeError("audit HTTP forbidden")):
                session=cached_session(fastf1,y,r);raw=pd.DataFrame(session.laps)
        except Exception as exc:
            assert not receipt["available"] and receipt["error_type"]==type(exc).__name__
            continue
        assert receipt["available"] and receipt["raw_laps"]==len(raw)
        assert not raw.duplicated(["Driver","LapNumber"]).any()
        timelines=defaultdict(list)
        for row in raw.itertuples():
            if pd.notna(row.Time) and pd.notna(row.Position):timelines[row.LapNumber].append(row)
        selected=[]
        for n,rows in timelines.items():
            rows=sorted(rows,key=lambda x:(x.Position,x.Driver))
            previous=None
            for row in rows:
                clear=row.Position==1 or (previous is not None and (row.Time-previous).total_seconds()>1.5)
                previous=row.Time
                if not clear or n<=1 or pd.isna(row.LapTime) or pd.notna(row.PitInTime) or pd.notna(row.PitOutTime):continue
                if str(row.TrackStatus)!="1" or pd.isna(row.IsAccurate) or not row.IsAccurate:continue
                if row.Compound not in ("INTERMEDIATE","WET") or pd.isna(row.TyreLife) or not math.isfinite(row.TyreLife) or row.TyreLife<0:continue
                seconds=row.LapTime.total_seconds()
                if not math.isfinite(seconds) or seconds<=0:continue
                selected.append((row,seconds))
        grouped=defaultdict(list)
        for row,seconds in selected:grouped[(row.Driver,row.Compound)].append(seconds)
        typical={key:median(v) for key,v in grouped.items()}
        eligible={(row.Driver,int(row.LapNumber),row.Compound):row for row,seconds in selected if seconds<1.07*typical[(row.Driver,row.Compound)]}
        roster=entries[entries.season.eq(y)&entries["round"].eq(r)]
        assert {key[0] for key in eligible}.issubset(set(roster.code))
        current=[]
        for team,g in roster.groupby("team",sort=True):
            codes=sorted(g.code)
            if len(codes)!=2:continue
            a,b=codes
            for (d,n,c),left in eligible.items():
                if d!=a or (b,n,c) not in eligible:continue
                right=eligible[(b,n,c)]
                if abs(left.TyreLife-right.TyreLife)<=3:
                    current.append(dict(team=team,a=a,b=b,lap=n,compound=c,age_a=float(left.TyreLife),age_b=float(right.TyreLife),season=y,round=r))
        assert receipt["matched_laps"]==len(current)
        source.extend(current)
        print("audited",y,r,len(current),flush=True)
    expected=pd.DataFrame(source,columns=saved.columns)
    keys=["season","round","team","lap"]
    pd.testing.assert_frame_equal(saved.sort_values(keys).reset_index(drop=True),expected.sort_values(keys).reset_index(drop=True),check_dtype=False,atol=0,rtol=0)
    groups=defaultdict(list)
    for row in source:groups[(row["season"],row["round"],row["team"],row["a"],row["b"])].append(row["lap"])
    pairs=[]
    for (y,r,t,a,b),laps in sorted(groups.items()):
        even=sum(n%2==0 for n in laps);odd=len(laps)-even
        pairs.append(dict(season=y,round=r,team=t,a=a,b=b,n=len(laps),half0=even,half1=odd,supported=len(laps)>=10 and even>=5 and odd>=5))
    pd.testing.assert_frame_equal(table,pd.DataFrame(pairs,columns=table.columns),check_dtype=False)
    good=[p for p in pairs if p["supported"]]
    races={(p["season"],p["round"]) for p in good};seasons={p["season"] for p in good}
    drivers=Counter(d for p in good for d in (p["a"],p["b"]))
    repeated=sum(n>=3 for n in drivers.values())
    numbers=dict(matched_laps=len(source),pair_events=len(pairs),supported_pairs=len(good),supported_races=len(races),supported_seasons=len(seasons),drivers_three_wet_events=repeated)
    assert all(report[k]==v for k,v in numbers.items())
    gate=len(good)>=40 and len(races)>=8 and len(seasons)>=3 and repeated>=8
    assert report["gate"]==gate and not report["winner_fit"] and not report["response_fit"]
    assert len(report["sessions"])==103 and len({(r["season"],r["round"]) for r in report["sessions"]})==103
    assert set((r["season"],r["round"]) for r in report["sessions"])==set(map(tuple,entries[["season","round"]].drop_duplicates().to_numpy()))
    assert not any(k in saved for k in ("LapTime","gap","pace","winner","finish"))
    print("E246 all103raw-event scalar traffic/conditional medians/pair/parity/driver/support/no-response audit PASS")


if __name__=="__main__":main()
