"""E238 bounded native DriverRaceInfo physical-field observability pilot."""
import gzip
import hashlib
import json
import os
import re
from unittest.mock import patch

import requests

from experiments.e190_traffic_feasibility import PILOT,cached_session
from experiments.e196_weather_source import restricted_send
from experiments.e230_native_score_schema import stream_profile
from f1.data import ROOT

FIELDS={'fuel','fuelload','fuellevel','fuelremaining','fuelconsumption','energy','ersenergy','erscharge',
    'batterycharge','stateofcharge','enginetemperature','watertemperature','oiltemperature',
    'tyretemperature','tiretemperature','tyrepressure','tirepressure','steering','steeringangle',
    'brakepressure','gforce','longitudinalg','lateralg','lateralacceleration','longitudinalacceleration'}


def feed_url(path):
    if not path.startswith('/static/2019/') or not path.endswith('_Race/'):
        raise ValueError('unregistered source year/session')
    return 'https://livetiming.formula1.com'+path+'DriverRaceInfo.jsonStream'


def physical_profile(raw,roster):
    summary,records=stream_profile(raw,roster)
    leaves=[]; cars=set(); paths=set()
    for record in records:
        for leaf in record['numeric']:
            if any(re.sub('[^a-z0-9]','',name.lower()) in FIELDS for name in leaf['path']):
                leaves.append(leaf['value'])
                cars.update(name for name in leaf['path'] if name in roster)
                paths.add(tuple('*' if name.isdigit() else name for name in leaf['path']))
    quality=bool(records and summary['malformed_records']/(len(records)+summary['malformed_records'])<=.05)
    summary.update(physical_leaves=len(leaves),physical_car_keys=sorted(cars),physical_paths=[list(p) for p in sorted(paths)],
        physical_qualified=bool(quality and len(leaves)>=100 and len(cars)>=10 and len(set(leaves))>=2))
    return summary,records


def main():
    import fastf1
    import f1.ff1
    from fastf1.req import Cache
    from fastf1.internals.f1account import get_token
    reports=[]
    if not all(os.environ.get(k) for k in ('F1_USERNAME','F1_PASSWORD')):
        raise RuntimeError('Existing-account credentials unavailable')
    fastf1.Cache.offline_mode(False)
    try:
        get_token(method='password')
    except Exception as exc:
        (ROOT/'results/e238_driver_race_schema.json').write_text(json.dumps(dict(sources=[],gate=False,error_type=type(exc).__name__,winner_fit=False))+'\n')
        return
    for season,round_ in PILOT:
        fastf1.Cache.offline_mode(True)
        session=cached_session(fastf1,season,round_)
        roster=set(session.laps.DriverNumber.dropna().astype(str))
        row=dict(season=season,round=round_,available=False,physical_qualified=False,requests=[])
        try:
            url=feed_url(session.api_path); row['url']=url
            fastf1.Cache.offline_mode(False)
            with patch('requests.adapters.HTTPAdapter.send',new=restricted_send(requests.adapters.HTTPAdapter.send,{url},row['requests'])),Cache.disabled():
                with Cache.requests_get(url,headers=fastf1._api.headers,timeout=20,stream=True,allow_redirects=False) as response:
                    row.update(status=response.status_code,last_modified=response.headers.get('Last-Modified'),http_date=response.headers.get('Date'))
                    if response.status_code!=200:
                        raise ValueError('native source unavailable')
                    raw=response.raw.read(4*1024*1024+1)
                    if len(raw)>4*1024*1024:
                        raise ValueError('native source exceeds bound')
            summary,records=physical_profile(raw,roster)
            row.update(available=True,raw_bytes=len(raw),raw_sha256=hashlib.sha256(raw).hexdigest(),**summary)
            (ROOT/f'results/e238_2019_{round_}_numeric.json.gz').write_bytes(gzip.compress((json.dumps(records,indent=2)+'\n').encode(),mtime=0))
        except Exception as exc:
            row['error_type']=type(exc).__name__
        finally:
            fastf1.Cache.offline_mode(True)
        reports.append(row)
        print('SOURCE',round_,row.get('status'),row.get('records',0),row.get('physical_leaves',0),row['physical_qualified'],flush=True)
    result=dict(sources=reports,gate=sum(r['physical_qualified'] for r in reports)>=2,winner_fit=False,physical_meanings_verified=False)
    (ROOT/'results/e238_driver_race_schema.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
