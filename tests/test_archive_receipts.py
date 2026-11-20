from urllib.parse import urlsplit,parse_qs
from experiments.e252_archive_receipts import parse,query_url,FIELDS


def main():
    original="https://livetiming.formula1.com/static/2019/race/SPFeed.jsonStream"
    row=["20191101000000",original,"200","text/plain","A"*32]
    assert parse([FIELDS,row],original,"2019-05-12")[0]["plausible_metadata"]
    assert parse([],original,"2019-05-12")==[] and parse([FIELDS],original,"2019-05-12")==[]
    for index,value in [(0,"20200101000000"),(0,"20190511000000"),(1,original+"extra"),(2,"403"),(3,"text/html"),(4,"-")]:
        changed=row.copy();changed[index]=value
        assert not parse([FIELDS,changed],original,"2019-05-12")[0]["plausible_metadata"]
    params=parse_qs(urlsplit(query_url(original)).query)
    assert params["url"]==[original] and params["matchType"]==["exact"] and params["limit"]==["100"]
    assert params["to"]==["20191231235959"]
    try:parse([["incorrect"]],original,"2019-05-12")
    except AssertionError:pass
    else:raise AssertionError("invalid header accepted")
    import json,tempfile
    from pathlib import Path
    from unittest.mock import patch
    import experiments.e252_archive_receipts as module
    class Response:
        status_code=403
        def __init__(self,url):self.url=url
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def iter_content(self,chunk_size):yield b"x"*65536;yield b"x"
    class Session:
        def __init__(self):self.calls=0
        def get(self,url,**kwargs):self.calls+=1;return Response(url)
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);(root/"results").mkdir()
        sources=[dict(season=2019,round=r,feed="SPFeed",url=original) for r in (5,12,14)]
        (root/"results/e230_native_schema.json").write_text(json.dumps(dict(sources=sources)))
        session=Session()
        with patch.object(module,"ROOT",root),patch.object(module.requests,"Session",return_value=session):module.main()
        report=json.loads((root/"results/e252_archive_receipts.json").read_text())
        assert session.calls==1 and report["host_denial"] and report["complete"]
        assert report["sources"][0]["error_type"]=="ValueError"
        assert not any(r["attempted"] for r in report["sources"][1:])
    print("E252 exactURL/time/MIME/status/digest/empty/header/query-bound checks PASS")


if __name__=="__main__":main()
