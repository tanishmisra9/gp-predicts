"""Different decoder and raw HTTP cache access for E259 provenance."""
import hashlib
import inspect
import json
import math
from collections import defaultdict
from unittest.mock import patch
import pandas as pd
from experiments.e190_traffic_feasibility import cached_session
from f1.data import ROOT


def main():
    import fastf1
    import fastf1._api
    import f1.ff1
    fastf1.Cache.offline_mode(True)
    samples=pd.read_csv(ROOT/"results/e259_packet_checks.csv",float_precision="round_trip",dtype={"car":str})
    eps=pd.read_csv(ROOT/"results/e259_episode_checks.csv",float_precision="round_trip")
    original=pd.read_csv(ROOT/"results/e258_episodes.csv",float_precision="round_trip")
    report=json.loads((ROOT/"results/e259_summary.json").read_text())
    old=json.loads((ROOT/"results/e247_native_channel_inventory.json").read_text())
    assert report["parser_sha256"]==hashlib.sha256(inspect.getsource(fastf1._api.car_data).encode()).hexdigest()
    channels={"0":"RPM","2":"Speed","3":"nGear","4":"Throttle","5":"Brake"}
    for receipt,earlier in zip(report["receipts"],old["sources"]):
        assert (receipt["season"],receipt["round"])==(earlier["season"],earlier["round"])
        current=samples[samples["round"].eq(receipt["round"])]
        wanted={(r.car,pd.Timestamp(r.date).value) for r in current.itertuples()}
        urls=tuple(base+earlier["path"]+"CarData.z.jsonStream" for base in (fastf1._api.base_url,fastf1._api.base_url_mirror))
        with patch("requests.adapters.HTTPAdapter.send",side_effect=RuntimeError("audit HTTP forbidden")):
            response=fastf1.Cache.requests_get(urls,headers=fastf1._api.headers)
            session=cached_session(fastf1,2019,receipt["round"])
            cars=fastf1._api.car_data(session.api_path)
        assert response.status_code==200 and response.from_cache and response.url in urls
        text=response.content.decode("utf-8-sig")
        assert hashlib.sha256(text.encode()).hexdigest()==receipt["record_stream_sha256"]==earlier["record_stream_sha256"]
        assert text.endswith("\r\n")
        lines=text.split("\r\n")[:-1]; assert len(lines)==receipt["records"]
        found=defaultdict(list)
        for i,line in enumerate(lines):
            obj=fastf1._api.parse(line[12:],zipped=True)
            for j,entry in enumerate(obj["Entries"]):
                date=pd.Timestamp(entry["Utc"]).tz_convert(None).floor("us")
                for car,state in entry["Cars"].items():
                    key=(car,date.value)
                    if key in wanted: found[key].append((i,j,line[:12],entry["Utc"],state["Channels"]))
        assert len(found)==receipt["found"]==receipt["wanted"]==len(wanted)
        for row in current.itertuples():
            key=(row.car,pd.Timestamp(row.date).value)
            i,j,elapsed,utc,raw=found[key][-1]
            assert (row.record_index,row.entry_index,row.elapsed,row.utc,row.occurrences)==(i,j,elapsed,utc,len(found[key]))
            parsed=cars[row.car].loc[lambda d:d.Date.notna()].drop_duplicates("Date",keep="last").sort_values("Date").iloc[row.native_index]
            assert parsed.Date==pd.Timestamp(row.date) and parsed.Time==pd.to_timedelta(elapsed)
            for c,name in channels.items():
                assert c in raw and type(raw[c]) in (int,float) and math.isfinite(raw[c])
                assert getattr(row,"raw_"+name)==raw[c]
                assert getattr(row,"parsed_"+name)==parsed[name]==(bool(raw[c]) if name=="Brake" else raw[c])
            event=original.loc[row.event_id]
            assert row.driver==event.driver and row.round==event["round"]
            assert row.core==(event.first_index<=row.native_index<=event.last_index)
            assert row.all_explicit and row.values_equal and row.time_equal and row.observed_time
        print("native provenance audited",receipt["round"],len(current),flush=True)
    for event_id,event in original.iterrows():
        g=samples[samples.event_id.eq(event_id)].sort_values("native_index")
        assert list(g.native_index)==list(range(int(event.first_index)-1,int(event.last_index)+2))
        core=g[g.core]; e=eps[eps.event_id.eq(event_id)].iloc[0]
        assert (e.samples,e.distinct_rpm,e.distinct_speed,e.distinct_tuples)==(len(core),core.parsed_RPM.nunique(),core.parsed_Speed.nunique(),len(core[["parsed_"+n for n in channels.values()]].drop_duplicates()))
    passed=(len(samples)==31 and samples.core.sum()==25 and len(eps)==3 and eps.distinct_rpm.ge(3).all() and eps.distinct_speed.ge(3).all())
    assert report["gate"]==bool(passed) and not report["physical_fault_verified"] and not report["winner_fit"]
    for name,digest in report["files"].items():assert hashlib.sha256((ROOT/"results"/name).read_bytes()).hexdigest()==digest
    print("E259 HTTP-cache/FastF1-decoder/UTC-floor/identity/index/value/time/variation/hash/gate audit PASS")


if __name__=="__main__":main()
