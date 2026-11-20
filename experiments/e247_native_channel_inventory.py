"""E247 native compressed car channel inventory, cache only, no semantics/fit."""
import hashlib
import json
import math
import re
from unittest.mock import patch

from experiments.e190_traffic_feasibility import cached_session
from f1.data import ROOT

PILOT=((2019,5),(2019,12),(2019,14))
KNOWN={"0","2","3","4","5","45"}


def inventory(records,roster,decode):
    assert len(roster)>=10
    stats={};entries=0;updates=0
    for text in records:
        assert isinstance(text,str) and len(text)>12
        payload=decode(text[12:])
        assert isinstance(payload,dict) and isinstance(payload["Entries"],list)
        for entry in payload["Entries"]:
            assert isinstance(entry["Cars"],dict)
            entries+=1
            for car,state in entry["Cars"].items():
                assert car in roster and isinstance(state["Channels"],dict)
                updates+=1
                for channel,value in state["Channels"].items():
                    assert isinstance(channel,str) and re.fullmatch(r"[0-9]{1,4}",channel)
                    s=stats.setdefault((car,channel),dict(count=0,numeric=0,nonnumeric=0,minimum=None,maximum=None))
                    s["count"]+=1
                    if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):
                        s["numeric"]+=1
                        s["minimum"]=value if s["minimum"] is None else min(s["minimum"],value)
                        s["maximum"]=value if s["maximum"] is None else max(s["maximum"],value)
                    else:s["nonnumeric"]+=1
    rows=[dict(car=car,channel=channel,**s) for (car,channel),s in sorted(stats.items())]
    extra=sorted({r["channel"] for r in rows}-KNOWN)
    supported={c:sum(r["channel"]==c and r["numeric"]>=100 and r["minimum"]<r["maximum"] for r in rows) for c in extra}
    return dict(records=len(records),entries=entries,car_updates=updates,channels=sorted({r["channel"] for r in rows}),
                extra_channels=extra,supported_extra_cars=supported,rows=rows,
                gate=bool(len(records)>=100 and any(n>=10 for n in supported.values())))


def main():
    import fastf1
    import fastf1._api
    import f1.ff1
    fastf1.Cache.offline_mode(True);reports=[]
    for year,round_ in PILOT:
        report=dict(season=year,round=round_,available=False,gate=False)
        try:
            with patch("requests.adapters.HTTPAdapter.send",side_effect=RuntimeError("native channel HTTP forbidden")):
                session=cached_session(fastf1,year,round_)
                roster=sorted(set(session.laps.DriverNumber.dropna().astype(str)))
                assert "/2019/" in session.api_path and session.api_path.endswith("Race/")
                records=fastf1._api.fetch_page(session.api_path,"car_data")
        except Exception as exc:report["error_type"]=type(exc).__name__
        else:
            if records is None:report["reason"]="raw_feed_not_cached"
            else:
                result=inventory(records,set(roster),lambda s:fastf1._api.parse(s,zipped=True))
                digest=hashlib.sha256()
                for record in records:digest.update(record.encode());digest.update(b"\r\n")
                report.update(available=True,path=session.api_path,roster=roster,
                              record_stream_sha256=digest.hexdigest(),**result)
        reports.append(report)
        print("SOURCE",json.dumps({k:v for k,v in report.items() if k!="rows"}),flush=True)
    summary=dict(sources=reports,gate=bool(all(r["available"] for r in reports) and sum(r["gate"] for r in reports)>=2),
                 semantics_verified=False,winner_fit=False)
    (ROOT/"results/e247_native_channel_inventory.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("NOVEL_CHANNEL_GATE",summary["gate"],flush=True)


if __name__=="__main__":main()
