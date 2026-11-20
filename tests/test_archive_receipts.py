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
    print("E252 exactURL/time/MIME/status/digest/empty/header/query-bound checks PASS")


if __name__=="__main__":main()
