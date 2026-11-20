"""E259 native packet provenance; values are not mechanical fault labels."""
import base64
import hashlib
import inspect
import json
import math
import re
import zlib
from datetime import datetime
from unittest.mock import patch

import pandas as pd

from experiments.e190_traffic_feasibility import cached_session
from f1.data import ROOT

CHANNELS = {"0":"RPM", "2":"Speed", "3":"nGear", "4":"Throttle", "5":"Brake"}


def decode(body):
    obj = json.loads(body)
    if isinstance(obj, str):
        inflater = zlib.decompressobj(-15)
        payload = inflater.decompress(base64.b64decode(obj, validate=True)) + inflater.flush()
        assert inflater.eof and not inflater.unused_data and not inflater.unconsumed_tail
        obj = json.loads(payload)
    assert isinstance(obj, dict) and isinstance(obj["Entries"], list)
    return obj


def lookup(records, wanted):
    result = {}
    for record_index, line in enumerate(records):
        assert re.fullmatch(r"[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}", line[:12])
        for entry_index, entry in enumerate(decode(line[12:])["Entries"]):
            # datetime's ISO parser truncates the native seventh fractional digit.
            date = pd.Timestamp(datetime.fromisoformat(entry["Utc"].removesuffix("Z")))
            for car, state in entry["Cars"].items():
                key = (car, date.value)
                if key not in wanted: continue
                channels = state["Channels"]
                previous = result.get(key)
                result[key] = dict(record_index=record_index, entry_index=entry_index, utc=entry["Utc"],
                                   elapsed=line[:12], occurrences=1 if previous is None else previous["occurrences"]+1,
                                   channels={c:channels[c] for c in CHANNELS if c in channels})
    return result


def compare(parsed, raw):
    explicit = raw is not None and all(c in raw["channels"] and type(raw["channels"][c]) in (int,float)
                                      and math.isfinite(raw["channels"][c]) for c in CHANNELS)
    same = bool(explicit and all(getattr(parsed,name) == (bool(raw["channels"][c]) if name=="Brake" else raw["channels"][c])
                                for c,name in CHANNELS.items()))
    time = bool(raw is not None and pd.notna(parsed.Time) and parsed.Time == pd.to_timedelta(raw["elapsed"]))
    return dict(all_explicit=bool(explicit), values_equal=same, time_equal=time, observed_time=bool(pd.notna(parsed.Time)))


def gate(samples, episodes):
    return bool(len(samples)==31 and int(samples.core.sum())==25 and len(episodes)==3
                and samples[["all_explicit","values_equal","time_equal","observed_time"]].all().all()
                and episodes.distinct_rpm.ge(3).all() and episodes.distinct_speed.ge(3).all())


def main():
    import fastf1
    import fastf1._api
    import f1.ff1
    fastf1.Cache.offline_mode(True)
    old = json.loads((ROOT/"results/e247_native_channel_inventory.json").read_text())
    source = pd.read_csv(ROOT/"results/e258_episodes.csv",float_precision="round_trip")
    assert len(source)==3
    rows, summaries, receipts = [], [], []
    for receipt in old["sources"]:
        year, round_ = receipt["season"], receipt["round"]
        assert (year,round_) in ((2019,5),(2019,12),(2019,14))
        with patch("requests.adapters.HTTPAdapter.send",side_effect=RuntimeError("E259 HTTP forbidden")):
            session = cached_session(fastf1,year,round_)
            laps = pd.DataFrame(session.laps)
            cars = fastf1._api.car_data(session.api_path)
            records = fastf1._api.fetch_page(session.api_path,"car_data")
        digest = hashlib.sha256()
        for line in records: digest.update(line.encode()); digest.update(b"\r\n")
        assert digest.hexdigest()==receipt["record_stream_sha256"]
        wanted, selected = set(), []
        for event_id, event in source[source["round"].eq(round_)].iterrows():
            numbers = laps.loc[laps.Driver.eq(event.driver),"DriverNumber"].dropna().unique()
            assert len(numbers)==1
            number = str(numbers[0])
            d = cars[number].loc[lambda x:x.Date.notna()].drop_duplicates("Date",keep="last").sort_values("Date").reset_index(drop=True)
            for index in range(int(event.first_index)-1,int(event.last_index)+2):
                parsed = next(d.iloc[[index]].itertuples())
                wanted.add((number,parsed.Date.value))
                selected.append((event_id,event,number,index,parsed))
        raw = lookup(records,wanted)
        for event_id,event,number,index,parsed in selected:
            native = raw.get((number,parsed.Date.value))
            checked = compare(parsed,native)
            row = dict(event_id=int(event_id),season=year,round=round_,driver=event.driver,car=number,
                       native_index=index,core=bool(event.first_index<=index<=event.last_index),date=parsed.Date.isoformat(),**checked)
            row.update({"parsed_"+name:getattr(parsed,name) for name in CHANNELS.values()})
            row.update({"raw_"+name:None if native is None else native["channels"].get(c) for c,name in CHANNELS.items()})
            row.update({k:None if native is None else native[k] for k in ("record_index","entry_index","utc","elapsed","occurrences")})
            rows.append(row)
        receipts.append(dict(season=year,round=round_,record_stream_sha256=digest.hexdigest(),records=len(records),wanted=len(wanted),found=len(raw)))
    samples = pd.DataFrame(rows)
    for event_id,g in samples[samples.core].groupby("event_id",sort=True):
        summaries.append(dict(event_id=int(event_id),samples=len(g),distinct_rpm=int(g.parsed_RPM.nunique()),
                              distinct_speed=int(g.parsed_Speed.nunique()),distinct_tuples=len(g[["parsed_"+c for c in CHANNELS.values()]].drop_duplicates())))
    episodes = pd.DataFrame(summaries)
    files = {}
    for name,frame in [("packet_checks",samples),("episode_checks",episodes)]:
        path = ROOT/f"results/e259_{name}.csv";frame.to_csv(path,index=False)
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(gate=gate(samples,episodes),samples=len(samples),core_samples=int(samples.core.sum()),receipts=receipts,
                  parser_sha256=hashlib.sha256(inspect.getsource(fastf1._api.car_data).encode()).hexdigest(),files=files,
                  physical_fault_verified=False,winner_fit=False)
    (ROOT/"results/e259_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
