"""E247 direct raw HTTP-cache/decompression/statistics audit, no fetch_page."""
import base64
import hashlib
import json
import math
import re
import zlib
from collections import defaultdict
from unittest.mock import patch

from f1.data import ROOT


def main():
    import fastf1
    import fastf1._api
    import f1.ff1
    report=json.loads((ROOT/"results/e247_native_channel_inventory.json").read_text())
    assert [(r["season"],r["round"]) for r in report["sources"]]==[(2019,5),(2019,12),(2019,14)]
    fastf1.Cache.offline_mode(True);qualified=0
    known={"0","2","3","4","5","45"}
    for receipt in report["sources"]:
        assert receipt["available"] and "/2019/" in receipt["path"] and receipt["path"].endswith("Race/")
        urls=tuple(base+receipt["path"]+"CarData.z.jsonStream" for base in (fastf1._api.base_url,fastf1._api.base_url_mirror))
        with patch("requests.adapters.HTTPAdapter.send",side_effect=RuntimeError("audit HTTP forbidden")):
            response=fastf1.Cache.requests_get(urls,headers=fastf1._api.headers)
        assert response.status_code==200 and response.from_cache and response.url in urls
        text=response.content.decode("utf-8-sig")
        assert text.endswith("\r\n")
        assert hashlib.sha256(text.encode()).hexdigest()==receipt["record_stream_sha256"]
        lines=text.split("\r\n")[:-1]
        stats=defaultdict(lambda:dict(count=0,numeric=0,nonnumeric=0,minimum=None,maximum=None))
        entries=0;updates=0
        for line in lines:
            assert re.fullmatch(r"[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}",line[:12])
            outer=json.loads(line[12:])
            if isinstance(outer,str):
                decoder=zlib.decompressobj(-15)
                data=decoder.decompress(base64.b64decode(outer))+decoder.flush()
                assert decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
                payload=json.loads(data.decode("utf-8-sig"))
            else:payload=outer
            assert isinstance(payload,dict) and isinstance(payload["Entries"],list)
            entries+=len(payload["Entries"])
            for patch_ in payload["Entries"]:
                updates+=len(patch_["Cars"])
                for car,body in patch_["Cars"].items():
                    assert car in receipt["roster"]
                    for channel,value in body["Channels"].items():
                        assert re.fullmatch(r"[0-9]{1,4}",channel)
                        s=stats[(car,channel)];s["count"]+=1
                        if type(value) in (int,float) and math.isfinite(value):
                            s["numeric"]+=1
                            if s["minimum"] is None or value<s["minimum"]:s["minimum"]=value
                            if s["maximum"] is None or value>s["maximum"]:s["maximum"]=value
                        else:s["nonnumeric"]+=1
        expected=[dict(car=car,channel=channel,**s) for (car,channel),s in sorted(stats.items())]
        assert expected==receipt["rows"]
        assert (len(lines),entries,updates)==(receipt["records"],receipt["entries"],receipt["car_updates"])
        channels=sorted({c for _,c in stats});extra=sorted(set(channels)-known)
        counts={c:sum(s["numeric"]>=100 and s["minimum"]<s["maximum"] for (car,ch),s in stats.items() if ch==c) for c in extra}
        gate=len(lines)>=100 and any(n>=10 for n in counts.values())
        assert (channels,extra,counts,gate)==(receipt["channels"],receipt["extra_channels"],receipt["supported_extra_cars"],receipt["gate"])
        qualified+=int(gate)
        print("audited",receipt["season"],receipt["round"],len(lines),updates,channels,flush=True)
    assert report["gate"]==(qualified>=2)
    assert not report["semantics_verified"] and not report["winner_fit"]
    print("E247 direct cached raw HTTP framing/hash/independent deflate decode/all counts/minmax/schema/gate audit PASS")


if __name__=="__main__":main()
