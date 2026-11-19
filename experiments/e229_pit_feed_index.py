"""Three historical session-index metadata requests; no pit payload, audio or text fields."""
import json
import os
from unittest.mock import patch

import requests

from experiments.e190_traffic_feasibility import PILOT,cached_session
from experiments.e196_weather_source import restricted_send
from f1.data import ROOT

CANDIDATES={'PitStopSeries','PitStopTime','PitStopData'}


def index_url(path):
    if not path.startswith('/static/2019/') or not path.endswith('_Race/'):
        raise ValueError('non-pilot-year/Race endpoint')
    return 'https://livetiming.formula1.com'+path+'Index.json'


def feed_names(payload):
    feeds=payload.get('Feeds')
    if not isinstance(feeds,dict) or len(feeds)>500 or not all(isinstance(k,str) for k in feeds):
        raise ValueError('unknown feed-index schema')
    names=sorted(feeds)
    return names,sorted(set(names)&CANDIDATES)


def main():
    import fastf1
    import f1.ff1
    from fastf1.req import Cache
    from fastf1.internals.f1account import get_token
    if not all(os.environ.get(k) for k in ('F1_USERNAME','F1_PASSWORD')):
        result=dict(sessions=[],gate=False,error_type='MissingExistingCredentials')
        (ROOT/'results/e229_pit_feed_index.json').write_text(json.dumps(result,indent=2)+'\n');return
    fastf1.Cache.offline_mode(False)
    try:get_token(method='password')
    except Exception as exc:
        result=dict(sessions=[],gate=False,error_type=type(exc).__name__)
        (ROOT/'results/e229_pit_feed_index.json').write_text(json.dumps(result,indent=2)+'\n');return
    rows=[]
    for season,rnd in PILOT:
        row=dict(season=season,round=rnd,available=False,requests=[])
        try:
            fastf1.Cache.offline_mode(True)
            session=cached_session(fastf1,season,rnd);url=index_url(session.api_path)
            row['url']=url
            fastf1.Cache.offline_mode(False)
            send=requests.adapters.HTTPAdapter.send
            with patch('requests.adapters.HTTPAdapter.send',new=restricted_send(send,{url},row['requests'])), Cache.disabled():
                with Cache.requests_get(url,headers=fastf1._api.headers,timeout=20,stream=True,allow_redirects=False) as response:
                    row['status']=response.status_code
                    if response.status_code!=200:raise ValueError('index unavailable')
                    payload=response.raw.read(2*1024*1024+1)
                    if len(payload)>2*1024*1024:raise ValueError('index exceeds bound')
                    names,candidates=feed_names(json.loads(payload.decode('utf-8-sig')))
                    row.update(available=True,feeds=names,pit_candidates=candidates)
        except Exception as exc:row['error_type']=type(exc).__name__
        finally:fastf1.Cache.offline_mode(True)
        rows.append(row)
        print('SOURCE',json.dumps(row),flush=True)
    passed=sum(r['available'] and bool(r['pit_candidates']) for r in rows)>=2
    result=dict(sessions=rows,gate=bool(passed),pit_payload_acquired=False,winner_fit=False)
    (ROOT/'results/e229_pit_feed_index.json').write_text(json.dumps(result,indent=2)+'\n')
    print('INDEX GATE',passed,flush=True)


if __name__=='__main__':main()
