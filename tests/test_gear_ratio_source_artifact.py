"""Independent scalar/native-clock reconstruction of E258 source artifacts."""
import hashlib
import json
import math
from collections import Counter, defaultdict
from statistics import median
from unittest.mock import patch

import pandas as pd

from experiments.e190_traffic_feasibility import cached_session
from f1.data import ROOT


def scalar(car, intervals, origin):
    records = list(car[car.Date.notna()].drop_duplicates("Date", keep="last").sort_values("Date").itertuples())
    candidates = []
    for i in range(1, len(records)-1):
        triple = records[i-1:i+2]
        if not all(pd.notna(r.Time) and pd.notna(r.RPM) and math.isfinite(r.RPM) and r.RPM >= 1000
                   and pd.notna(r.Speed) and math.isfinite(r.Speed) and r.Speed >= 180
                   and pd.notna(r.Throttle) and 98 <= r.Throttle <= 100
                   and pd.notna(r.Brake) and r.Brake == False
                   and pd.notna(r.nGear) and math.isfinite(r.nGear) and 4 <= r.nGear <= 8
                   and int(r.nGear) == r.nGear for r in triple): continue
        if len({r.nGear for r in triple}) != 1: continue
        if not all(0 < (b.Date-a.Date).value <= 500_000_000 for a,b in zip(triple,triple[1:])): continue
        matches = [lap for a,b,lap in intervals if a <= triple[0].Date and triple[-1].Date < b]
        assert len(matches) <= 1
        if not matches: continue
        r = records[i]
        candidates.append(dict(index=i, time=(r.Date-origin).total_seconds(), lap=matches[0], gear=int(r.nGear), ratio=r.RPM/r.Speed))
    groups = defaultdict(list)
    for r in candidates: groups[r["gear"]].append(r["ratio"])
    supported = {g:median(v) for g,v in groups.items() if len(v) >= 50}
    marked = []
    for r in candidates:
        if r["gear"] not in supported: continue
        residual = math.log(r["ratio"]/supported[r["gear"]])
        if abs(residual) > math.log(1.10): marked.append(dict(r, residual=residual, sign=1 if residual>0 else -1))
    runs = []
    for r in marked:
        if not runs or r["index"] != runs[-1][-1]["index"]+1 or any(r[k] != runs[-1][-1][k] for k in ("lap","gear","sign")):
            runs.append([])
        runs[-1].append(r)
    events=[]
    for run in runs:
        a,b=run[0],run[-1]
        if len(run) < 5 or b["time"]-a["time"] < 1: continue
        events.append(dict(lap=a["lap"],gear=a["gear"],sign=a["sign"],first_index=a["index"],last_index=b["index"],
                           first=a["time"],last=b["time"],samples=len(run),max_abs_log_ratio=max(abs(r["residual"]) for r in run)))
    return dict(native_samples=len(records),eligible=len(candidates),supported_gears=len(supported),marked=len(marked),episodes=len(events)),events


def main():
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True)
    support=pd.read_csv(ROOT/"results/e258_driver_support.csv",float_precision="round_trip")
    episodes=pd.read_csv(ROOT/"results/e258_episodes.csv",float_precision="round_trip")
    result=json.loads((ROOT/"results/e258_source_summary.json").read_text())
    counts=Counter(); all_events=[]
    for round_ in (5,12,14):
        with patch("requests.adapters.HTTPAdapter.send",side_effect=RuntimeError("audit HTTP forbidden")):
            s=cached_session(fastf1,2019,round_)
            laps=pd.DataFrame(s.laps); cars=fastf1._api.car_data(s.api_path)
        assert not laps.duplicated(["Driver","LapNumber"]).any()
        origin=max((c.Date-c.Time).max() for c in cars.values() if c.Time.notna().any()).round("ms")
        for driver, raw in laps.groupby("Driver",sort=True):
            intervals=[]
            for r in raw.itertuples():
                if r.LapNumber > 1 and pd.notna(r.LapTime) and r.LapTime > pd.Timedelta(0) and pd.notna(r.Time) and pd.notna(r.LapStartTime) and pd.isna(r.PitInTime) and pd.isna(r.PitOutTime) and str(r.TrackStatus)=="1":
                    intervals.append((origin+r.LapStartTime,origin+r.Time,int(r.LapNumber)))
            intervals.sort()
            assert all(a<b for a,b,_ in intervals) and all(b<=c for (_,b,_),(c,_,_) in zip(intervals,intervals[1:]))
            numbers=raw.DriverNumber.dropna().unique(); assert len(numbers)==1
            summary,events=scalar(cars[str(numbers[0])],intervals,origin)
            actual=support[(support["round"]==round_)&(support.driver==driver)]
            assert len(actual)==1 and actual.iloc[0].season==2019
            assert {k:int(actual.iloc[0][k]) for k in summary}==summary,(round_,driver,summary)
            if summary["eligible"]>=100 and summary["supported_gears"]>=2: counts[round_]+=1
            all_events.extend(dict(e,season=2019,round=round_,driver=driver) for e in events)
        print("audited pilot",round_,counts[round_],flush=True)
    expected=pd.DataFrame(all_events,columns=episodes.columns)
    keys=["season","round","driver","first_index"]
    pd.testing.assert_frame_equal(expected.sort_values(keys).reset_index(drop=True),episodes.sort_values(keys).reset_index(drop=True),
                                  check_dtype=False,atol=2e-6,rtol=1e-12)
    assert len(support)==60 and not support.duplicated(["season","round","driver"]).any()
    independently_passed=(all(counts[r]>=10 for r in (5,12,14)) and len({r["round"] for r in all_events})>=2 and len({r["driver"] for r in all_events})>=3)
    assert result["reports"]==[dict(season=2019,round=r,available=True) for r in (5,12,14)]
    assert result["gate"]==independently_passed and not result["winner_fit"]
    assert result["episodes"]==len(all_events) and result["episode_drivers"]==len({r["driver"] for r in all_events})
    for name,digest in result["files"].items(): assert hashlib.sha256((ROOT/"results"/name).read_bytes()).hexdigest()==digest
    print("E258 all60cars scalar triple/absolute lap clock/median/run/episode/support/hash/gate audit PASS")


if __name__=="__main__":main()
