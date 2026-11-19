"""Bounded numeric-only schema projection of two unmeasured 2019 native feeds."""
import hashlib
import json
import math
import os
import re
from collections import Counter,defaultdict
from unittest.mock import patch

import requests

from experiments.e190_traffic_feasibility import PILOT,cached_session
from experiments.e196_weather_source import restricted_send
from f1.data import ROOT

FEEDS=('DriverScore','SPFeed')
SKIP={'message','text','msg','comment','commentary','description','audio','path','status','recording_url','recordingurl'}
EXCLUDE={'racingnumber','drivernumber','lap','lapnumber','position','line','utc','time','timestamp'}
NUMBER=re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')
SAFE=re.compile(r'[A-Za-z0-9_.-]{1,64}')


def feed_url(path,feed):
    if feed not in FEEDS or not path.startswith('/static/2019/') or not path.endswith('_Race/'):
        raise ValueError('unregistered feed/year/session')
    return 'https://livetiming.formula1.com'+path+feed+'.jsonStream'


def project(payload,roster):
    numeric=[];types=Counter();candidates=set()
    def walk(node,path=()):
        if isinstance(node,dict):
            for key in node:
                if not isinstance(key,str):raise ValueError('non-string JSON key')
                if key.lower() in SKIP:
                    types['skipped_text_or_path']+=1;continue
                name=key if SAFE.fullmatch(key) else '<other-key>'
                if key in roster:candidates.add(key)
                value=node[key]
                if key in ('RacingNumber','DriverNumber') and isinstance(value,(str,int)) and str(value) in roster:
                    candidates.add(str(value))
                walk(value,path+(name,))
        elif isinstance(node,list):
            for value in node:walk(value,path+('[]',))
        elif isinstance(node,bool):types['boolean']+=1
        elif isinstance(node,(int,float)) or (isinstance(node,str) and len(node)<=64 and NUMBER.fullmatch(node)):
            value=float(node)
            if not math.isfinite(value):types['nonfinite']+=1;return
            kind='numeric_string' if isinstance(node,str) else 'number';types[kind]+=1
            numeric.append(dict(path=list(path),value=value))
        elif node is None:types['null']+=1
        else:types['string_or_other']+=1
    walk(payload)
    return numeric,dict(types),sorted(candidates)


def stream_profile(raw,roster):
    records=[];malformed=0
    for line in raw.decode('utf-8-sig').splitlines():
        if not line.strip():continue
        if len(line)<14 or not re.fullmatch(r'\d\d:\d\d:\d\d[.:]\d{3}',line[:12]):
            malformed+=1;continue
        clock=line[:12]
        hour,minute,second,milli=int(clock[:2]),int(clock[3:5]),int(clock[6:8]),int(clock[9:12])
        if minute>=60 or second>=60:malformed+=1;continue
        try:payload=json.loads(line[12:])
        except json.JSONDecodeError:malformed+=1;continue
        numeric,types,cars=project(payload,roster)
        records.append(dict(seconds=hour*3600+minute*60+second+milli/1000,numeric=numeric,types=types,candidate_car_keys=cars))
    paths=defaultdict(list);cars=set()
    for record in records:
        cars.update(record['candidate_car_keys'])
        for item in record['numeric']:
            path=tuple('*' if k.isdigit() else k for k in item['path'])
            paths[path].append(item['value'])
    profiles=[]
    for path,values in sorted(paths.items()):
        profiles.append(dict(path=list(path),count=len(values),minimum=min(values),maximum=max(values),
                             candidate_measurement=bool(path and path[-1].lower() not in EXCLUDE and len(values)>=10 and max(values)>min(values))))
    candidate=any(p['candidate_measurement'] for p in profiles)
    good=bool(records and malformed/(len(records)+malformed)<=.05 and len(cars)>=10 and candidate)
    summary=dict(records=len(records),malformed_records=malformed,numeric_leaves=sum(len(r['numeric']) for r in records),
                 candidate_car_keys=sorted(cars),profiles=profiles,qualified=good)
    return summary,records


def main():
    import fastf1
    import f1.ff1
    from fastf1.req import Cache
    from fastf1.internals.f1account import get_token
    if not all(os.environ.get(k) for k in ('F1_USERNAME','F1_PASSWORD')):
        result=dict(sources=[],gate=False,error_type='MissingExistingCredentials',winner_fit=False)
        (ROOT/'results/e230_native_schema.json').write_text(json.dumps(result,indent=2)+'\n');return
    fastf1.Cache.offline_mode(False)
    try:get_token(method='password')
    except Exception as exc:
        result=dict(sources=[],gate=False,error_type=type(exc).__name__,winner_fit=False)
        (ROOT/'results/e230_native_schema.json').write_text(json.dumps(result,indent=2)+'\n');return
    reports=[]
    for season,rnd in PILOT:
        fastf1.Cache.offline_mode(True);session=cached_session(fastf1,season,rnd)
        roster=set(session.laps.DriverNumber.dropna().astype(str))
        for feed in FEEDS:
            row=dict(season=season,round=rnd,feed=feed,available=False,qualified=False,requests=[])
            try:
                url=feed_url(session.api_path,feed);row['url']=url
                fastf1.Cache.offline_mode(False)
                send=requests.adapters.HTTPAdapter.send
                with patch('requests.adapters.HTTPAdapter.send',new=restricted_send(send,{url},row['requests'])),Cache.disabled():
                    with Cache.requests_get(url,headers=fastf1._api.headers,timeout=20,stream=True,allow_redirects=False) as response:
                        row['status']=response.status_code
                        if response.status_code!=200:raise ValueError('native stream unavailable')
                        raw=response.raw.read(4*1024*1024+1)
                        if len(raw)>4*1024*1024:raise ValueError('native stream exceeds bound')
                row['stage']='schema_projection'
                summary,records=stream_profile(raw,roster)
                row.update(available=True,raw_bytes=len(raw),raw_sha256=hashlib.sha256(raw).hexdigest(),**summary)
                path=ROOT/f'results/e230_{season}_{rnd}_{feed}_numeric.json'
                path.write_text(json.dumps(records,indent=2)+'\n')
            except Exception as exc:row['error_type']=type(exc).__name__
            finally:fastf1.Cache.offline_mode(True)
            reports.append(row)
            print('SOURCE',json.dumps(row),flush=True)
    good={feed:sum(r['feed']==feed and r['qualified'] for r in reports) for feed in FEEDS}
    result=dict(sources=reports,qualified_sessions=good,gate=bool(any(v>=2 for v in good.values())),winner_fit=False)
    (ROOT/'results/e230_native_schema.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SCHEMA GATE',result['gate'],flush=True)


if __name__=='__main__':main()
