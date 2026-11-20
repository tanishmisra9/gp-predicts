"""E250 existing SPFeed named-branch indexed update support, not score semantics."""
import gzip
import hashlib
import json
import math
from collections import Counter

from experiments.e249_spfeed_schema_evidence import FIELDS
from f1.data import ROOT,load_results

PILOT=((2019,5),(2019,12),(2019,14))


def analyse(records,roster):
    values={(f,c):[] for f in FIELDS for c in sorted(roster)}
    rejected=[];excluded=Counter();explicit=0;index1=0;invalid=0
    def reject(reason,i,t,item):
        rejected.append(dict(reason=reason,record=i,seconds=t,path=item["path"],value=item["value"]))
    for i,record in enumerate(records):
        t=record["seconds"]
        assert isinstance(t,(int,float)) and not isinstance(t,bool) and math.isfinite(t) and t>=0
        for item in record["numeric"]:
            p=item["path"];v=item["value"]
            assert isinstance(p,list) and all(isinstance(k,str) for k in p)
            assert isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
            if p[:2]!=["Scores","graph"]:continue
            if len(p)>=3 and p[2]=="TrackStatus":excluded["track_status"]+=1;continue
            if len(p)<3 or p[2] not in FIELDS:
                invalid+=1;reject("unknown_field",i,t,item);continue
            if len(p)!=5:
                invalid+=1;reject("unexpected_depth",i,t,item);continue
            field,car,coordinate=p[2:]
            if not car.startswith("p") or car[1:] not in roster:
                invalid+=1;reject("unknown_driver",i,t,item);continue
            if coordinate=="[]":reject("ambiguous_array_index",i,t,item);continue
            if not coordinate.isdigit():
                invalid+=1;reject("nonnumeric_coordinate",i,t,item);continue
            explicit+=1
            if coordinate!="1":reject("other_explicit_coordinate",i,t,item);continue
            index1+=1
            if v!=int(v):
                invalid+=1;reject("noninteger_value",i,t,item);continue
            values[(field,car[1:])].append((t,v))
    rows=[]
    for (field,car),pairs in sorted(values.items()):
        times={t for t,v in pairs};numbers={v for t,v in pairs}
        span=max(times)-min(times) if times else 0.
        rows.append(dict(field=field,driver=car,updates=len(pairs),unique_times=len(times),distinct_values=len(numbers),
                         minimum=min(numbers) if numbers else None,maximum=max(numbers) if numbers else None,
                         first=min(times) if times else None,last=max(times) if times else None,span=span,
                         supported=bool(len(times)>=10 and len(numbers)>=2 and span>=300)))
    fields={f:sum(r["field"]==f and r["supported"] for r in rows) for f in FIELDS}
    drivers={c:sum(r["driver"]==c and r["supported"] for r in rows) for c in sorted(roster)}
    fraction=index1/explicit if explicit else None
    gate=bool(invalid==0 and fraction is not None and fraction>=.99 and
              sum(n>=10 for n in fields.values())>=5 and sum(n>=5 for n in drivers.values())>=10)
    return dict(records=len(records),explicit_updates=explicit,index1_updates=index1,index1_fraction=fraction,
                invalid_updates=invalid,rejection_counts=dict(Counter(r["reason"] for r in rejected)),excluded_counts=dict(excluded),
                supported_drivers_by_field=fields,supported_fields_by_driver=drivers,rows=rows,rejections=rejected,gate=gate)


def main():
    entries=load_results().loc[lambda d:d.season.eq(2019),["season","round","code"]]
    old=json.loads((ROOT/"results/e230_native_schema.json").read_text())
    reports=[]
    for season,round_ in PILOT:
        path=ROOT/f"results/e230_{season}_{round_}_SPFeed_numeric.json.gz"
        records=json.loads(gzip.decompress(path.read_bytes()))
        roster=set(entries[entries["round"].eq(round_)].code)
        assert len(roster)==20
        reference=[r for r in old["sources"] if r["season"]==season and r["round"]==round_ and r["feed"]=="SPFeed"]
        assert len(reference)==1 and reference[0]["qualified"]
        result=analyse(records,roster)
        result.update(season=season,round=round_,projection_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      original_raw_sha256=reference[0]["raw_sha256"])
        reports.append(result)
        print("SUPPORT",json.dumps({k:v for k,v in result.items() if k not in ("rows","rejections")}),flush=True)
    result=dict(sources=reports,gate=sum(r["gate"] for r in reports)>=2,physical_units_verified=False,
                axis_meaning_verified=False,historical_publication_verified=False,winner_fit=False)
    (ROOT/"results/e250_spfeed_patch_support.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":main()
