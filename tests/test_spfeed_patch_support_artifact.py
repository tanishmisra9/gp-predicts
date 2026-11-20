"""E250 independently rebuild projected patch classifications and grouped stats."""
import gzip
import hashlib
import json
from collections import Counter

import pandas as pd

from f1.data import ROOT,load_results


def main():
    fields=("Steering","GforceLat","GforceLong","Brake","Performance","Throttle")
    report=json.loads((ROOT/"results/e250_spfeed_patch_support.json").read_text())
    old=json.loads((ROOT/"results/e230_native_schema.json").read_text())
    entries=load_results().loc[lambda d:d.season.eq(2019),["round","code"]]
    assert [(r["season"],r["round"]) for r in report["sources"]]==[(2019,5),(2019,12),(2019,14)]
    passed=0
    for saved in report["sources"]:
        season,round_=saved["season"],saved["round"]
        path=ROOT/f"results/e230_{season}_{round_}_SPFeed_numeric.json.gz"
        assert hashlib.sha256(path.read_bytes()).hexdigest()==saved["projection_sha256"]
        reference=next(r for r in old["sources"] if r["season"]==season and r["round"]==round_ and r["feed"]=="SPFeed")
        assert reference["raw_sha256"]==saved["original_raw_sha256"]
        roster=sorted(entries[entries["round"].eq(round_)].code)
        records=json.loads(gzip.decompress(path.read_bytes()))
        reject=[];good=[];excluded=Counter();explicit=0;coordinate1=0;invalid=0
        for index,r in enumerate(records):
            for item in r["numeric"]:
                p=item["path"];v=item["value"];reason=None
                if tuple(p[:2])!=("Scores","graph"):continue
                if len(p)>2 and p[2]=="TrackStatus":excluded["track_status"]+=1;continue
                if len(p)<3 or p[2] not in fields:reason="unknown_field"
                elif len(p)!=5:reason="unexpected_depth"
                elif p[3] not in {"p"+c for c in roster}:reason="unknown_driver"
                elif p[4]=="[]":reason="ambiguous_array_index"
                elif not p[4].isdigit():reason="nonnumeric_coordinate"
                else:
                    explicit+=1
                    if p[4]!="1":reason="other_explicit_coordinate"
                    else:
                        coordinate1+=1
                        if int(v)!=v:reason="noninteger_value"
                        else:good.append(dict(field=p[2],driver=p[3][1:],seconds=r["seconds"],value=v))
                if reason:
                    reject.append(dict(reason=reason,record=index,seconds=r["seconds"],path=p,value=v))
                    if reason not in ("ambiguous_array_index","other_explicit_coordinate"):invalid+=1
        assert reject==saved["rejections"]
        assert dict(Counter(r["reason"] for r in reject))==saved["rejection_counts"]
        assert dict(excluded)==saved["excluded_counts"]
        df=pd.DataFrame(good,columns=["field","driver","seconds","value"])
        expected=[]
        for field in sorted(fields):
            for driver in roster:
                g=df[df.field.eq(field)&df.driver.eq(driver)]
                n=int(g.seconds.nunique());distinct=int(g.value.nunique())
                first=float(g.seconds.min()) if len(g) else None
                last=float(g.seconds.max()) if len(g) else None
                span=last-first if len(g) else 0.
                expected.append(dict(field=field,driver=driver,updates=len(g),unique_times=n,distinct_values=distinct,
                    minimum=float(g.value.min()) if len(g) else None,maximum=float(g.value.max()) if len(g) else None,
                    first=first,last=last,span=span,supported=n>=10 and distinct>=2 and span>=300))
        assert expected==saved["rows"]
        fcounts={f:sum(r["field"]==f and r["supported"] for r in expected) for f in fields}
        dcounts={c:sum(r["driver"]==c and r["supported"] for r in expected) for c in roster}
        fraction=coordinate1/explicit if explicit else None
        gate=invalid==0 and fraction is not None and fraction>=.99 and sum(n>=10 for n in fcounts.values())>=5 and sum(n>=5 for n in dcounts.values())>=10
        assert [len(records),explicit,coordinate1,invalid,fraction]==[saved[k] for k in ("records","explicit_updates","index1_updates","invalid_updates","index1_fraction")]
        assert fcounts==saved["supported_drivers_by_field"] and dcounts==saved["supported_fields_by_driver"] and gate==saved["gate"]
        passed+=int(gate)
        print("audited",season,round_,len(expected),len(reject),gate)
    assert report["gate"]==(passed>=2)
    assert not any(report[k] for k in ("physical_units_verified","axis_meaning_verified","historical_publication_verified","winner_fit"))
    print("E250 independent all projection hashes/identity/path/rejection/groupedstats/coverage/gate audit PASS")


if __name__=="__main__":main()
