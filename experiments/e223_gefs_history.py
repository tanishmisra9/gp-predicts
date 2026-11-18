"""Registered 2018–22 local-window GEFS source acquisition; no winner evaluation."""
import csv
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from experiments.e220_gefs_metadata import before_cutoff
from experiments.e221_gefs_headers import identification, inventory
from experiments.e222_gefs_decode import KEYS

ROOT = Path(__file__).resolve().parents[1]
ZONES = {
 'albert_park':'Australia/Melbourne', 'americas':'America/Chicago', 'bahrain':'Asia/Bahrain',
 'baku':'Asia/Baku', 'catalunya':'Europe/Madrid', 'hockenheimring':'Europe/Berlin',
 'hungaroring':'Europe/Budapest', 'imola':'Europe/Rome', 'interlagos':'America/Sao_Paulo',
 'istanbul':'Europe/Istanbul', 'jeddah':'Asia/Riyadh', 'losail':'Asia/Qatar',
 'marina_bay':'Asia/Singapore', 'miami':'America/New_York', 'monaco':'Europe/Monaco',
 'monza':'Europe/Rome', 'mugello':'Europe/Rome', 'nurburgring':'Europe/Berlin',
 'portimao':'Europe/Lisbon', 'red_bull_ring':'Europe/Vienna', 'ricard':'Europe/Paris',
 'rodriguez':'America/Mexico_City', 'shanghai':'Asia/Shanghai', 'silverstone':'Europe/London',
 'sochi':'Europe/Moscow', 'spa':'Europe/Brussels', 'suzuka':'Asia/Tokyo',
 'villeneuve':'America/Toronto', 'yas_marina':'Asia/Dubai', 'zandvoort':'Europe/Amsterdam'}
NIGHT = {'bahrain','jeddah','losail','marina_bay','yas_marina'}


def race_plan(race, coords):
    season, rnd = int(race['season']), int(race['round'])
    if not 2018 <= season <= 2022:
        raise ValueError('source-year boundary')
    day = datetime.fromisoformat(race['date'])
    if day.year != season or day.weekday() != 6:
        raise ValueError('registered Sunday calendar expected')
    circuit = race['circuit']
    local = day.replace(hour=18 if circuit in NIGHT else 12, tzinfo=ZoneInfo(ZONES[circuit]))
    start = local.astimezone(timezone.utc); stop = (local + timedelta(hours=6)).astimezone(timezone.utc)
    cutoff = (day - timedelta(days=3)).replace(tzinfo=timezone.utc)
    init = cutoff - timedelta(hours=12)
    intervals = []
    for end in range(6, 169, 6):
        left, right = init + timedelta(hours=end-6), init + timedelta(hours=end)
        overlap = max(0., (min(stop, right) - max(start, left)).total_seconds() / 3600)
        if overlap:
            stamp = init.strftime('%Y%m%d')
            name = (f'atmos/pgrb2ap5/gec00.t12z.pgrb2a.0p50.f{end:03d}' if stamp >= '20200923'
                    else f'pgrb2a/gec00.t12z.pgrb2af{end}')
            intervals.append(dict(end=end, overlap_hours=overlap,
                                  url=f'https://noaa-gefs-pds.s3.amazonaws.com/gefs.{stamp}/12/{name}'))
    if not math.isclose(sum(i['overlap_hours'] for i in intervals), 6):
        raise ValueError('window not covered')
    return dict(season=season, round=rnd, date=race['date'], circuit=circuit,
                lat=float(coords[circuit]['lat']), lon=float(coords[circuit]['lon']),
                zone=ZONES[circuit], local_start=local.isoformat(), window_start=start.isoformat(),
                window_stop=stop.isoformat(), cutoff=cutoff.isoformat(), initialization=init.isoformat(),
                intervals=intervals)


def plans():
    with (ROOT / 'data/circuit_static.csv').open() as f:
        coords = {r['circuit']:r for r in csv.DictReader(f)}
    races = {}
    with (ROOT / 'data/results.csv').open() as f:
        for row in csv.DictReader(f):
            if 2018 <= int(row['season']) <= 2022:
                key = (int(row['season']), int(row['round']))
                item = {k:row[k] for k in ('season','round','date','circuit')}
                if key in races and races[key] != item:
                    raise ValueError('inconsistent calendar identity')
                races[key] = item
    return [race_plan(races[k], coords) for k in sorted(races)]


def decode_field(data, plan, end, variable):
    import eccodes as E
    ident = identification(data[:37])
    if (ident['reference'] != plan['initialization'] or ident['first_message_length'] != len(data)
            or ident['significance'] != 1 or ident['production'] != 0 or ident['data_type'] != 3
            or data[-4:] != b'7777'):
        raise ValueError('GRIB identity mismatch')
    handle = E.codes_new_from_message(data)
    try:
        m = {k:E.codes_get(handle,k) for k in KEYS}
        p = dict(E.codes_grib_find_nearest(handle,plan['lat'],plan['lon'],npoints=1)[0])
    finally:
        E.codes_release(handle)
    init = datetime.fromisoformat(plan['initialization']); valid = init + timedelta(hours=end)
    common = (m['edition']==2 and m['discipline']==0 and m['dataDate']==int(init.strftime('%Y%m%d'))
              and m['dataTime']==1200 and m['validityDate']==int(valid.strftime('%Y%m%d'))
              and m['validityTime']==int(valid.strftime('%H%M')) and str(m['stepUnits'])=='1'
              and m['endStep']==end and m['gridType']=='regular_ll'
              and math.isfinite(p['value']) and 0<=p['distance']<=80)
    if variable=='TMP':
        field = (m['parameterCategory']==0 and m['parameterNumber']==0 and m['units']=='K'
                 and m['typeOfLevel']=='heightAboveGround' and m['level']==2
                 and m['startStep']==end and m['stepType']=='instant' and 180<=p['value']<=330)
    else:
        field = (variable=='APCP' and m['parameterCategory']==1 and m['parameterNumber']==8
                 and m['units']=='kg m**-2' and m['typeOfLevel']=='surface'
                 and m['startStep']==end-6 and m['stepType']=='accum' and 0<=p['value']<=500)
    if not (common and field):
        raise ValueError('decoded field/time/unit/location gate failure')
    return dict(metadata=m, nearest=p, value=p['value']-273.15 if variable=='TMP' else p['value'])


def acquire(plan):
    import eccodes as E
    row = dict(plan=plan, objects=[], complete=False)
    try:
        for interval in plan['intervals']:
            url = interval['url']
            with requests.head(url,timeout=20,allow_redirects=False) as response:
                receipt = dict(status=response.status_code,content_length=response.headers.get('Content-Length'),
                               last_modified=response.headers.get('Last-Modified'),etag=response.headers.get('ETag'),cutoff=plan['cutoff'],
                               accessed=datetime.now(timezone.utc).isoformat())
            obj = dict(url=url,receipt=receipt,end=interval['end'],overlap_hours=interval['overlap_hours'],fields={})
            row['objects'].append(obj)
            if not before_cutoff(receipt) or not receipt['etag']:
                raise ValueError('uncertified source receipt')
            with requests.get(url+'.idx',timeout=20,stream=True,allow_redirects=False) as response:
                if response.status_code!=200: raise ValueError('index unavailable')
                payload=response.raw.read(1024*1024+1)
            if len(payload)>1024*1024: raise ValueError('index exceeds bound')
            inv=inventory(payload.decode('ascii'))
            obj['index']=inv
            if any(i['initialization']!=datetime.fromisoformat(plan['initialization']).strftime('%Y%m%d%H') for i in inv):
                raise ValueError('index reference mismatch')
            for variable,level in (('TMP','2 m above ground'),('APCP','surface')):
                selected=[i for i in inv if i['variable']==variable and i['level']==level]
                if len(selected)!=1: raise ValueError('ambiguous or missing field')
                item=selected[0]; number=item['number']; start=item['offset']
                end=inv[number]['offset']-1 if number<len(inv) else int(receipt['content_length'])-1
                size=end-start+1
                if not 37<=size<=1024*1024: raise ValueError('field exceeds bound')
                with requests.get(url,headers={'Range':f'bytes={start}-{end}','If-Match':receipt['etag'],
                                  'Accept-Encoding':'identity'},timeout=20,stream=True,allow_redirects=False) as response:
                    current=receipt | dict(last_modified=response.headers.get('Last-Modified'))
                    if (response.status_code!=206 or response.headers.get('ETag')!=receipt['etag']
                            or response.headers.get('Content-Range')!=f"bytes {start}-{end}/{receipt['content_length']}"
                            or not before_cutoff(current)):
                        raise ValueError('unconfirmed field range/identity/cutoff')
                    data=response.raw.read(size+1)
                if len(data)!=size: raise ValueError('message size mismatch')
                path=ROOT / f"data/parts/gefs_e223/{plan['season']}_{plan['round']:02d}_{interval['end']}_{variable}.grib2"
                path.write_bytes(data)
                value=decode_field(data,plan,interval['end'],variable)
                obj['fields'][variable]=value | dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest(),
                                                   start=start,end=end,last_modified=current['last_modified'])
        row['forecast_temp_c']=sum(o['fields']['TMP']['value']*o['overlap_hours']/6 for o in row['objects'])
        row['forecast_precip_mm']=sum(o['fields']['APCP']['value']*o['overlap_hours']/6 for o in row['objects'])
        row['complete']=True
    except (requests.RequestException,ValueError,OSError,E.CodesInternalError) as exc:
        row['error']=type(exc).__name__+': '+str(exc)
    return row


def main():
    schedule=plans()
    if len(schedule)!=103: raise ValueError('registered 103-race calendar changed')
    (ROOT/'data/parts/gefs_e223').mkdir(parents=True,exist_ok=True)
    directory=ROOT/'results/e223_receipts'; directory.mkdir(exist_ok=True)
    missing=[]
    for plan in schedule:
        path=directory / f"{plan['season']}_{plan['round']:02d}.json"
        if not path.exists(): missing.append(plan)
        elif json.loads(path.read_text())['plan']!=plan: raise ValueError('saved plan mismatch')
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks={pool.submit(acquire,p):p for p in missing}
        for task in as_completed(tasks):
            row=task.result(); p=row['plan']
            (directory/f"{p['season']}_{p['round']:02d}.json").write_text(json.dumps(row,indent=2)+'\n')
            print(p['season'],p['round'],'complete' if row['complete'] else row.get('error'),flush=True)
    rows=[json.loads((directory/f"{p['season']}_{p['round']:02d}.json").read_text()) for p in schedule]
    summary={str(y):dict(total=sum(r['plan']['season']==y for r in rows),
                        complete=sum(r['plan']['season']==y and r['complete'] for r in rows)) for y in range(2018,2023)}
    passed=(sum(r['complete'] for r in rows)/103>=.9 and all(s['complete']/s['total']>=.8 for s in summary.values()))
    (ROOT/'results/e223_gefs_coverage.json').write_text(json.dumps(dict(years=summary,source_gate=passed,
          winner_utility_tested=False,forecast_approximation='local six-hour window; uniform interval rainfall; endpoint temperature'),indent=2)+'\n')
    print(json.dumps(summary), 'source gate',passed,flush=True)


if __name__=='__main__': main()
