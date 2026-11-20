"""E252 bounded public CDX metadata lookup; no archive content or model."""
import gzip
import hashlib
import json
import re
from datetime import datetime
from urllib.parse import urlsplit

import requests

from f1.data import ROOT

FIELDS=["timestamp","original","statuscode","mimetype","digest"]
MIME={"application/json","text/plain","application/octet-stream","application/x-ndjson"}


def query_url(original):
    return requests.Request("GET","https://web.archive.org/cdx/search/cdx",params={
        "url":original,"matchType":"exact","output":"json","fl":",".join(FIELDS),
        "from":"20190101000000","to":"20191231235959","filter":"statuscode:200","limit":"100"}).prepare().url


def parse(payload,original,event_date):
    assert isinstance(payload,list)
    if not payload:return []
    assert payload[0]==FIELDS and len(payload)<=101
    target=urlsplit(original);rows=[]
    for values in payload[1:]:
        assert isinstance(values,list) and len(values)==len(FIELDS) and all(isinstance(v,str) for v in values)
        r=dict(zip(FIELDS,values));stamp=r["timestamp"]
        assert re.fullmatch(r"[0-9]{14}",stamp)
        captured=datetime.strptime(stamp,"%Y%m%d%H%M%S")
        url=urlsplit(r["original"])
        exact=url.scheme in ("http","https") and url.netloc==target.netloc and url.path==target.path and url.query==target.query
        within=datetime.fromisoformat(event_date)<=captured<datetime(2020,1,1)
        digest=bool(re.fullmatch(r"[A-Z2-7]{32}",r["digest"]))
        r["plausible_metadata"]=bool(exact and within and r["statuscode"]=="200" and r["mimetype"] in MIME and digest)
        rows.append(r)
    return rows


def main():
    destination=ROOT/"results/e252_archive_receipts.json"
    if destination.exists():
        print("Recorded bounded lookup already exists; no requests repeated")
        return
    old=json.loads((ROOT/"results/e230_native_schema.json").read_text())
    sources=[r for r in old["sources"] if r["feed"]=="SPFeed"]
    assert [(r["season"],r["round"]) for r in sources]==[(2019,5),(2019,12),(2019,14)]
    dates={5:"2019-05-12",12:"2019-08-04",14:"2019-09-08"}
    session=requests.Session();reports=[];denied=False
    for source in sources:
        url=query_url(source["url"])
        r=dict(season=2019,round=source["round"],original=source["url"],query=url,attempted=False,available=False,rows=[])
        if denied:r["reason"]="not_attempted_after_host_denial"
        else:
            r["attempted"]=True
            try:
                with session.get(url,timeout=20,allow_redirects=False,stream=True) as response:
                    assert response.url==url
                    r["status"]=response.status_code
                    if response.status_code in (401,403,429):denied=True
                    chunks=[];size=0
                    for chunk in response.iter_content(chunk_size=4096):
                        size+=len(chunk)
                        if size>65536:raise ValueError("archive metadata response exceeds64KiB")
                        chunks.append(chunk)
                    body=b"".join(chunks)
                    r.update(bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
                    path=ROOT/f"results/e252_2019_{source['round']}_cdx_response.bin.gz"
                    path.write_bytes(gzip.compress(body,mtime=0))
                    if response.status_code==200:
                        r["rows"]=parse(json.loads(body),source["url"],dates[source["round"]])
                        r["available"]=True;r["limit_reached"]=len(r["rows"])==100
                    else:r["reason"]="HTTP_non200_not_capture_absence"
            except Exception as exc:r["error_type"]=type(exc).__name__
        reports.append(r)
        result=dict(sources=reports,complete=len(reports)==3,host_denial=denied,
                    plausible_capture_metadata=sum(any(x["plausible_metadata"] for x in s["rows"]) for s in reports),
                    archive_body_acquired=False,raw_historical_publication_verified=False,winner_fit=False)
        destination.write_text(json.dumps(result,indent=2)+"\n")
        print("SOURCE",json.dumps({k:v for k,v in r.items() if k not in ("rows","query")}),flush=True)


if __name__=="__main__":main()
