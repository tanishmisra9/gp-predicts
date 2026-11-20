import base64
import json
import zlib
from types import SimpleNamespace
import pandas as pd
from experiments.e259_packet_audit import decode,lookup,compare,gate


def main():
    stamp="2019-01-01T00:00:01.1234567Z"
    channels={"0":10000,"2":250,"3":6,"4":100,"5":0}
    obj={"Entries":[{"Utc":stamp,"Cars":{"1":{"Channels":channels}}}]}
    compressor=zlib.compressobj(wbits=-15)
    zipped=compressor.compress(json.dumps(obj).encode())+compressor.flush()
    body=json.dumps(base64.b64encode(zipped).decode())
    assert decode(body)==obj and decode(json.dumps(obj))==obj
    key=("1",pd.Timestamp("2019-01-01T00:00:01.123456").value)
    record="00:00:01.000"+body
    result=lookup([record,record],{key})
    raw=result[key]
    assert raw["occurrences"]==2 and raw["record_index"]==1 and raw["entry_index"]==0
    parsed=SimpleNamespace(Time=pd.Timedelta(seconds=1),RPM=10000,Speed=250,nGear=6,Throttle=100,Brake=False)
    assert all(compare(parsed,raw).values())
    assert not compare(parsed,None)["all_explicit"]
    for field in channels:
        changed=dict(raw,channels={c:v for c,v in channels.items() if c!=field})
        assert not compare(parsed,changed)["all_explicit"]
    assert not compare(parsed,dict(raw,channels=dict(channels,**{"0":True})))["all_explicit"]
    assert not compare(parsed,dict(raw,channels=dict(channels,**{"2":251})))["values_equal"]
    assert not compare(parsed,dict(raw,elapsed="00:00:02.000"))["time_equal"]
    assert lookup([],{key})=={}
    try:decode(json.dumps(base64.b64encode(zipped[:-3]).decode()))
    except (AssertionError,zlib.error):pass
    else:raise AssertionError("truncated deflate accepted")
    samples=pd.DataFrame({c:[True]*31 for c in ("all_explicit","values_equal","time_equal","observed_time")})
    samples["core"]=[True]*25+[False]*6
    episodes=pd.DataFrame(dict(distinct_rpm=[3]*3,distinct_speed=[3]*3))
    assert gate(samples,episodes)
    assert not gate(samples,episodes.assign(distinct_speed=1))
    assert not gate(samples.assign(values_equal=False),episodes)
    assert not gate(samples.iloc[:-1],episodes)
    print("E259 explicit/missing/duplicate/altered/time/UTC/empty/deflate/heldtuple/gate tests PASS")


if __name__=="__main__":main()
