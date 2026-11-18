"""Fixed common-member rainfall ensemble acquisition, 2019–22; no winner model."""
import csv
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime,timedelta
from pathlib import Path
from threading import Lock

import requests

from experiments.e220_gefs_metadata import before_cutoff
from experiments.e221_gefs_headers import identification,inventory
from experiments.e222_gefs_decode import KEYS

ROOT=Path(__file__).resolve().parents[1]
DECODE_LOCK=Lock()


def member_url(control,member):
    if not 1<=member<=20 or control.count('gec00')!=1:
        raise ValueError('unregistered member or control URL')
    return control.replace('gec00',f'gep{member:02d}')


def field_gate(m,p,plan,end,member):
    init=datetime.fromisoformat(plan['initialization']); valid=init+timedelta(hours=end)
    return bool(m['edition']==2 and m['discipline']==0 and m['parameterCategory']==1 and m['parameterNumber']==8
        and m['units']=='kg m**-2' and m['typeOfLevel']=='surface'
        and m['dataDate']==int(init.strftime('%Y%m%d')) and m['dataTime']==1200
        and m['validityDate']==int(valid.strftime('%Y%m%d')) and m['validityTime']==int(valid.strftime('%H%M'))
        and str(m['stepUnits'])=='1' and m['startStep']==end-6 and m['endStep']==end and m['stepType']=='accum'
        and m['gridType']=='regular_ll' and m['perturbationNumber']==member
        and math.isfinite(p['value']) and 0<=p['value']<=500 and 0<=p['distance']<=80)


def decode(data,plan,end,member):
    import eccodes as E
    ident=identification(data[:37])
    if (ident['reference']!=plan['initialization'] or ident['first_message_length']!=len(data)
        or ident['significance']!=1 or ident['production']!=0 or ident['data_type']!=4 or data[-4:]!=b'7777'):
        raise ValueError('perturbed forecast identity mismatch')
    with DECODE_LOCK:
        handle=E.codes_new_from_message(data)
        try:
            m={k:E.codes_get(handle,k) for k in KEYS+('perturbationNumber','numberOfForecastsInEnsemble')}
            p=dict(E.codes_grib_find_nearest(handle,plan['lat'],plan['lon'],npoints=1)[0])
        finally:E.codes_release(handle)
    if not field_gate(m,p,plan,end,member): raise ValueError('field/time/member/unit/location mismatch')
    return dict(metadata=m,nearest=p,value=p['value'])


def acquire(plan,member):
    import eccodes as E
    row=dict(plan=plan,member=member,objects=[],complete=False)
    try:
        with requests.Session() as session:
            for interval in plan['intervals']:
                url=member_url(interval['url'],member)
                with session.head(url,timeout=20,allow_redirects=False) as response:
                    receipt=dict(status=response.status_code,content_length=response.headers.get('Content-Length'),
                        last_modified=response.headers.get('Last-Modified'),etag=response.headers.get('ETag'),
                        cutoff=plan['cutoff'],accessed=datetime.now().astimezone().isoformat())
                obj=dict(url=url,end=interval['end'],overlap_hours=interval['overlap_hours'],receipt=receipt)
                row['objects'].append(obj)
                if not before_cutoff(receipt) or not receipt['etag']: raise ValueError('uncertified member receipt')
                with session.get(url+'.idx',timeout=20,stream=True,allow_redirects=False) as response:
                    if response.status_code!=200: raise ValueError('member index unavailable')
                    payload=response.raw.read(1024*1024+1)
                if len(payload)>1024*1024: raise ValueError('member index exceeds bound')
                inv=inventory(payload.decode('ascii'))
                stamp=datetime.fromisoformat(plan['initialization']).strftime('%Y%m%d%H')
                if any(r['initialization']!=stamp for r in inv): raise ValueError('index reference mismatch')
                matches=[r for r in inv if r['variable']=='APCP' and r['level']=='surface']
                if len(matches)!=1: raise ValueError('ambiguous or missing APCP')
                item=matches[0]; number=item['number']; start=item['offset']
                end=inv[number]['offset']-1 if number<len(inv) else int(receipt['content_length'])-1
                size=end-start+1
                if not 37<=size<=1024*1024: raise ValueError('member field exceeds bound')
                obj.update(index_record=item,next_index_record=inv[number] if number<len(inv) else None,
                           start=start,end=end,index_sha256=hashlib.sha256(payload).hexdigest())
                with session.get(url,headers={'Range':f'bytes={start}-{end}','If-Match':receipt['etag'],
                     'Accept-Encoding':'identity'},timeout=20,stream=True,allow_redirects=False) as response:
                    current=receipt|dict(last_modified=response.headers.get('Last-Modified'))
                    obj['range_response']=dict(status=response.status_code,etag=response.headers.get('ETag'),
                                                content_range=response.headers.get('Content-Range'))
                    if (response.status_code!=206 or response.headers.get('ETag')!=receipt['etag']
                        or response.headers.get('Content-Range')!=f"bytes {start}-{end}/{receipt['content_length']}"
                        or not before_cutoff(current)): raise ValueError('unconfirmed member range/object/cutoff')
                    data=response.raw.read(size+1)
                if len(data)!=size: raise ValueError('member message size mismatch')
                path=ROOT/f"data/parts/gefs_e226/{plan['season']}_{plan['round']:02d}_{member:02d}_{interval['end']}.grib2"
                path.write_bytes(data)
                obj.update(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest(),last_modified=current['last_modified'])
                obj.update(decode(data,plan,interval['end'],member))
        row['forecast_rain']=sum(o['value']*o['overlap_hours']/6 for o in row['objects'])
        row['complete']=True
    except (requests.RequestException,ValueError,OSError,E.CodesInternalError) as exc:
        row['error']=type(exc).__name__+': '+str(exc)
    return row


def probability(control,members):
    if len(members)!=20 or not all(r['complete'] for r in members):
        return float(control>1),False
    assert sorted(r['member'] for r in members)==list(range(1,21))
    return (int(control>1)+sum(r['forecast_rain']>1 for r in members))/21,True


def main():
    sources=[json.loads(p.read_text()) for p in sorted((ROOT/'results/e223_receipts').glob('*.json'))]
    sources=[r for r in sources if r['plan']['season']>=2019]
    assert len(sources)==82 and all(r['complete'] for r in sources)
    assert all(2019<=r['plan']['season']<=2022 for r in sources)
    assert len({(r['plan']['season'],r['plan']['round']) for r in sources})==82
    assert sum(len(r['plan']['intervals']) for r in sources)==160
    directory=ROOT/'results/e226_receipts'; directory.mkdir(exist_ok=True)
    (ROOT/'data/parts/gefs_e226').mkdir(parents=True,exist_ok=True)
    tasks=[]
    for source in sources:
        plan=source['plan']
        for member in range(1,21):
            path=directory/f"{plan['season']}_{plan['round']:02d}_{member:02d}.json"
            if not path.exists():tasks.append((plan,member))
            else:
                row=json.loads(path.read_text()); assert row['plan']==plan and row['member']==member
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(acquire,p,m) for p,m in tasks]
        for n,task in enumerate(as_completed(futures),1):
            row=task.result(); p=row['plan']; m=row['member']
            (directory/f"{p['season']}_{p['round']:02d}_{m:02d}.json").write_text(json.dumps(row,indent=2)+'\n')
            if n%20==0:print('SOURCE',n,'/',len(tasks),p['season'],p['round'],m,'ok' if row['complete'] else row.get('error'),flush=True)
    windows=[]
    for source in sources:
        p=source['plan']; members=[json.loads((directory/f"{p['season']}_{p['round']:02d}_{m:02d}.json").read_text()) for m in range(1,21)]
        prob,complete=probability(source['forecast_precip_mm'],members)
        windows.append(dict(season=p['season'],round=p['round'],p_wet=prob,complete=complete,
                            control=float(source['forecast_precip_mm']>1),complete_members=sum(r['complete'] for r in members)))
    with (ROOT/'results/e226_ensemble_windows.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(windows[0]));writer.writeheader();writer.writerows(windows)
    coverage={str(y):dict(total=sum(r['season']==y for r in windows),complete=sum(r['season']==y and r['complete'] for r in windows)) for y in range(2019,2023)}
    gate=sum(r['complete'] for r in windows)>=78 and all(v['complete']/v['total']>=.8 for v in coverage.values())
    (ROOT/'results/e226_source_summary.json').write_text(json.dumps(dict(coverage=coverage,source_gate=gate),indent=2)+'\n')
    print('SOURCE GATE',gate,coverage,flush=True)


if __name__=='__main__':main()
