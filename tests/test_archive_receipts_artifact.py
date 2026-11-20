"""E252 independently inspect bounded lookup receipts and retained responses."""
import gzip
import hashlib
import json
import re
from datetime import datetime
from unittest.mock import patch
from urllib.parse import urlsplit,parse_qs

from f1.data import ROOT


def main():
    report=json.loads((ROOT/"results/e252_archive_receipts.json").read_text())
    old=json.loads((ROOT/"results/e230_native_schema.json").read_text())
    source=[s for s in old["sources"] if s["feed"]=="SPFeed"]
    assert len(report["sources"])==3 and report["complete"]
    blocked=False;captures=0
    fields=["timestamp","original","statuscode","mimetype","digest"]
    dates={5:datetime(2019,5,12),12:datetime(2019,8,4),14:datetime(2019,9,8)}
    for receipt,original in zip(report["sources"],source):
        assert (receipt["season"],receipt["round"],receipt["original"])==(original["season"],original["round"],original["url"])
        parsed=urlsplit(receipt["query"])
        assert parsed.scheme=="https" and parsed.netloc=="web.archive.org" and parsed.path=="/cdx/search/cdx"
        expected=dict(url=[original["url"]],matchType=["exact"],output=["json"],fl=[",".join(fields)],
                      **{"from":["20190101000000"],"to":["20191231235959"],"filter":["statuscode:200"],"limit":["100"]})
        assert parse_qs(parsed.query)==expected
        if blocked:
            assert not receipt["attempted"] and receipt["reason"]=="not_attempted_after_host_denial"
            continue
        assert receipt["attempted"]
        if receipt.get("status") in (401,403,429):blocked=True
        if "sha256" not in receipt:
            assert not receipt["available"] and "error_type" in receipt;continue
        body=gzip.decompress((ROOT/f"results/e252_2019_{receipt['round']}_cdx_response.bin.gz").read_bytes())
        assert len(body)==receipt["bytes"] and len(body)<=65536
        assert hashlib.sha256(body).hexdigest()==receipt["sha256"]
        if receipt["status"]!=200:
            assert not receipt["available"] and receipt["rows"]==[];continue
        payload=json.loads(body)
        assert receipt["available"] and isinstance(payload,list)
        if payload:assert payload[0]==fields and len(payload)<=101
        expected_rows=[]
        for values in payload[1:]:
            assert len(values)==5 and all(isinstance(v,str) for v in values)
            row=dict(zip(fields,values));timestamp=datetime.strptime(row["timestamp"],"%Y%m%d%H%M%S")
            a=urlsplit(row["original"]);b=urlsplit(original["url"])
            row["plausible_metadata"]=bool(a.scheme in ("http","https") and (a.netloc,a.path,a.query)==(b.netloc,b.path,b.query) and
                dates[receipt["round"]]<=timestamp<datetime(2020,1,1) and row["statuscode"]=="200" and
                row["mimetype"] in ("application/json","text/plain","application/octet-stream","application/x-ndjson") and
                re.fullmatch(r"[A-Z2-7]{32}",row["digest"]))
            expected_rows.append(row)
        assert receipt["rows"]==expected_rows and receipt["limit_reached"]==(len(expected_rows)==100)
        captures+=int(any(row["plausible_metadata"] for row in expected_rows))
    assert captures==report["plausible_capture_metadata"] and blocked==report["host_denial"]
    assert not any(report[k] for k in ("archive_body_acquired","raw_historical_publication_verified","winner_fit"))
    import experiments.e252_archive_receipts as module
    with patch.object(module.requests,"Session",side_effect=AssertionError("must not repeat lookup")):module.main()
    print("E252 independent exact scope/parameters/body bytes/hash/metadata/denial/claim and no-repeat audit PASS")


if __name__=="__main__":main()
