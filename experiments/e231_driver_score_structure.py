"""Same-body DriverScore structural/provenance audit; channels remain anonymous."""
import gzip
import hashlib
import json
import math
import os
from datetime import datetime,timedelta,timezone
from email.utils import parsedate_to_datetime
from unittest.mock import patch

import requests
import pandas as pd

from experiments.e190_traffic_feasibility import PILOT,cached_session
from experiments.e196_weather_source import restricted_send
from experiments.e230_native_score_schema import NUMBER,SKIP,feed_url
from f1.data import ROOT


def skeleton(value):
    if isinstance(value,dict):return {k:skeleton(value[k]) for k in value if k.lower() not in SKIP}
    if isinstance(value,list):return [skeleton(v) for v in value]
    if isinstance(value,bool) or value is None:return None
    if isinstance(value,(int,float)) or isinstance(value,str) and len(value)<=64 and NUMBER.fullmatch(value):
        number=float(value);return number if math.isfinite(number) else None
    return None


def indexed(value):
    if value is None:return []
    if isinstance(value,list):return list(enumerate(value))
    if isinstance(value,dict) and all(k.isdecimal() for k in value):return [(int(k),v) for k,v in value.items()]
    raise ValueError('unsupported indexed container')


def decode(raw,roster):
    records=[];observations=[];malformed=0;unknown=set()
    for record_number,line in enumerate(raw.decode('utf-8-sig').splitlines()):
        if not line.strip():continue
        if len(line)<14:malformed+=1;continue
        clock=line[:12]
        try:
            if clock[2]!=':' or clock[5]!=':' or clock[8] not in '.:':raise ValueError('clock')
            hour,minute,second,milli=int(clock[:2]),int(clock[3:5]),int(clock[6:8]),int(clock[9:12])
            if minute>=60 or second>=60:raise ValueError('clock range')
            payload=json.loads(line[12:]);scores=payload.get('Scores',{})
            if not isinstance(scores,dict):raise ValueError('Scores map')
        except (ValueError,TypeError,AttributeError):malformed+=1;continue
        seconds=hour*3600+minute*60+second+milli/1000
        clean=skeleton(scores);records.append(dict(record=record_number,seconds=seconds,scores=clean))
        for car,rows in clean.items():
            if car not in roster:unknown.add(car);continue
            for row_index,components in indexed(rows):
                for component,value in indexed(components):
                    if isinstance(value,(int,float)):
                        observations.append(dict(record=record_number,seconds=seconds,car=car,row=row_index,component=component,value=value))
    return records,observations,dict(records=len(records),malformed=malformed,unknown_car_keys=sorted(unknown))


def structure(observations,lap_rows):
    from collections import defaultdict
    history=defaultdict(list)
    for row in observations:history[(row['car'],row['row'],row['component'])].append(row)
    known={(str(r['car']),int(r['lap'])):float(r['time']) for r in lap_rows}
    axis={};components=defaultdict(set);ranges=defaultdict(list);timing=[];last_timing=[];revisions=0;drivers=defaultdict(set)
    for (car,row,component),updates in history.items():
        values={v['value'] for v in updates};revisions+=int(len(values)>1)
        if component==0:axis[(car,row)]=updates[-1]['value']
        else:
            components[component].add((car,row));ranges[component].append(updates[-1]['value'])
    variable={c for c,v in ranges.items() if max(v)>min(v)}
    matches=0
    for (car,row),lap in axis.items():
        good=lap==row+1 and (car,int(lap)) in known
        matches+=int(good)
        if not good:continue
        present=[]
        for component,pairs in components.items():
            if component not in variable:continue
            if (car,row) in pairs:
                present.append(component)
                timing.append(history[(car,row,component)][0]['seconds']-known[(car,int(lap))])
                last_timing.append(history[(car,row,component)][-1]['seconds']-known[(car,int(lap))])
        if len(present)>=2:drivers[car].add(row)
    covered=sum(len(rows)>=10 for rows in drivers.values())
    fraction=matches/len(axis) if axis else 0.
    clock_fraction=sum(t>=-2 for t in timing)/len(timing) if timing else 0.
    return dict(axis_rows=len(axis),axis_matches=matches,axis_match_fraction=fraction,
                nonaxis_components=sorted(components),variable_components=sorted(variable),
                component_ranges={str(c):[min(v),max(v)] for c,v in ranges.items()},drivers_with_10_rows_2_components=covered,
                first_measurement_delays=timing,last_measurement_delays=last_timing,postlap_fraction=clock_fraction,
                revised_coordinate_count=revisions,numeric_coordinates=len(history),
                structure_gate=bool(len(axis)>=100 and fraction>=.95 and covered>=10 and len(variable)>=2))


def main():
    import fastf1
    import f1.ff1
    from fastf1.req import Cache
    from fastf1.internals.f1account import get_token
    previous=json.loads((ROOT/'results/e230_native_schema.json').read_text())
    prior={(r['season'],r['round']):r for r in previous['sources'] if r['feed']=='DriverScore'}
    if not all(os.environ.get(k) for k in ('F1_USERNAME','F1_PASSWORD')):raise RuntimeError('MissingExistingCredentials')
    fastf1.Cache.offline_mode(False)
    try:get_token(method='password')
    except Exception as exc:
        (ROOT/'results/e231_structure_summary.json').write_text(json.dumps(dict(sources=[],gate=False,error_type=type(exc).__name__,winner_fit=False))+'\n');return
    reports=[]
    for season,rnd in PILOT:
        report=dict(season=season,round=rnd,available=False,requests=[])
        try:
            fastf1.Cache.offline_mode(True);session=cached_session(fastf1,season,rnd)
            roster=set(session.laps.DriverNumber.dropna().astype(str));url=feed_url(session.api_path,'DriverScore')
            report['url']=url;fastf1.Cache.offline_mode(False)
            send=requests.adapters.HTTPAdapter.send
            with patch('requests.adapters.HTTPAdapter.send',new=restricted_send(send,{url},report['requests'])),Cache.disabled():
                with Cache.requests_get(url,headers=fastf1._api.headers,timeout=20,stream=True,allow_redirects=False) as response:
                    report.update(status=response.status_code,last_modified=response.headers.get('Last-Modified'),etag=response.headers.get('ETag'),http_date=response.headers.get('Date'))
                    if response.status_code!=200:raise ValueError('DriverScore unavailable')
                    raw=response.raw.read(4*1024*1024+1)
                    if len(raw)>4*1024*1024:raise ValueError('source exceeds bound')
            digest=hashlib.sha256(raw).hexdigest();report['raw_sha256']=digest
            if digest!=prior[(season,rnd)]['raw_sha256']:raise ValueError('body differs from E230')
            records,observed,metadata=decode(raw,roster)
            laps=[dict(car=str(r.DriverNumber),lap=int(r.LapNumber),time=r.Time.total_seconds())
                  for r in session.laps.itertuples() if pd.notna(r.DriverNumber) and pd.notna(r.LapNumber)
                  and math.isfinite(r.LapNumber) and r.LapNumber>=1 and pd.notna(r.Time)]
            result=structure(observed,laps)
            # Conservative publication receipt: Thursday immediately after the historical race.
            race_date=datetime.strptime(session.api_path.split('/')[-2][:10],'%Y-%m-%d').replace(tzinfo=timezone.utc)
            limit=race_date+timedelta(days=4)
            try:
                modified=parsedate_to_datetime(report['last_modified'])
                timely=modified.tzinfo is not None and modified<=limit
            except (TypeError,ValueError):timely=False
            report.update(available=True,**metadata,**result,publication_limit=limit.isoformat(),publication_gate=timely)
            path=ROOT/f'results/e231_{season}_{rnd}_indexed.json.gz'
            content=dict(records=records,observations=observed,laps=laps)
            path.write_bytes(gzip.compress((json.dumps(content,indent=2)+'\n').encode(),mtime=0))
        except Exception as exc:report['error_type']=type(exc).__name__
        finally:fastf1.Cache.offline_mode(True)
        reports.append(report)
        print('SOURCE',json.dumps({k:v for k,v in report.items() if k not in ('first_measurement_delays','last_measurement_delays')}),flush=True)
    good=sum(r.get('available') and r.get('structure_gate') and r.get('publication_gate') and not r.get('unknown_car_keys') for r in reports)
    summary=dict(sources=reports,gate=bool(good>=2),winner_fit=False,channel_names_verified=False)
    (ROOT/'results/e231_structure_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('STRUCTURE/PROVENANCE GATE',summary['gate'],flush=True)


if __name__=='__main__':main()
