import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from experiments.e264_position_source import acquire,allowed_urls,FLOOR


def main():
    inventory=[dict(season=2018+i//21,round=i%21+1,available=i>=4) for i in range(103)]
    # The frozen-count mock need not model a real calendar, only a unique bounded cohort.
    assert inventory[-1]["season"]==2022
    calls=[];auth=[];mode={"offline":True}
    def cache(value):mode["offline"]=value
    class Sender:
        @staticmethod
        def send(adapter,request,**kw):calls.append(request.url);return SimpleNamespace(status_code=403)
    def positions(path):
        if mode["offline"]:raise RuntimeError("cache miss")
        request=SimpleNamespace(url=sorted(allowed_urls(path,int(path.split('/')[2])))[0])
        response=Sender.send(None,request)
        if response.status_code!=200:raise RuntimeError("denied")
        return {"1":1}
    def session(year,round_):return SimpleNamespace(api_path=f"/static/{year}/{round_}_Race/")
    with tempfile.TemporaryDirectory() as d:
        status=Path(d)/"status.json"
        r=acquire(inventory,status,session,positions,cache,lambda:auth.append(1),Sender,lambda:FLOOR+1)
        assert len(r)==103 and len(calls)==len(auth)==1
        assert sum(len(x["requests"]) for x in r)==1 and not any(x["requests"] for x in r[:4])
        assert all(x["stop_remaining"] for x in r[4:])
        before=status.read_bytes();acquire(inventory,status,session,positions,cache,lambda:auth.append(1),Sender,lambda:FLOOR+1)
        assert status.read_bytes()==before and len(calls)==1 and len(auth)==1
    for path,year in (("/static/2026/1_Race/",2026),("/static/2019/1_Qualifying/",2019)):
        try:allowed_urls(path,year)
        except AssertionError:pass
        else:raise AssertionError("off-cohort endpoint accepted")
    print("E264 bounded cohort/cache-first/one-denial-stop/auth/atomic checkpoint/no-repeat tests PASS")


if __name__=="__main__":main()
