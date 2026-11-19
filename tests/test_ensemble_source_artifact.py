"""Offline all-member raw GRIB/receipt/coordinate/frequency audit; isolated weather Python."""
import csv,hashlib,json,math,struct
from datetime import datetime,timedelta,timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from unittest.mock import patch


def test_saved_source():
    import numpy as np
    import eccodes as E
    root=Path(__file__).resolve().parents[1]
    receipts=[json.loads(p.read_text()) for p in sorted((root/'results/e226_receipts').glob('*.json'))]
    assert len(receipts)==1640
    control={}
    for path in (root/'results/e223_receipts').glob('*.json'):
        r=json.loads(path.read_text());p=r['plan']
        if p['season']>=2019:control[(p['season'],p['round'])]=r
    groups={};coordinates={};payloads=bytes_=0;paths=set()
    with patch('requests.sessions.Session.request',side_effect=AssertionError('audit must stay offline')):
        for r in receipts:
            p=r['plan'];key=p['season'],p['round'];member=r['member']
            assert 1<=member<=20 and p==control[key]['plan']
            groups.setdefault(key,{})[member]=r
            if not r['complete']:
                assert 'forecast_rain' not in r and 'error' in r
                continue
            assert len(r['objects'])==len(p['intervals'])
            init=datetime.fromisoformat(p['initialization']);cutoff=datetime.fromisoformat(p['cutoff'])
            total=0.
            for o,interval in zip(r['objects'],p['intervals']):
                assert o['url']==interval['url'].replace('gec00',f'gep{member:02d}')
                assert o['metadata']['endStep']==interval['end'] and o['overlap_hours']==interval['overlap_hours']
                h=o['receipt'];reply=o['range_response']
                assert h['status']==200 and int(h['content_length'])>0
                assert init<parsedate_to_datetime(h['last_modified'])<=cutoff
                assert init<parsedate_to_datetime(o['last_modified'])<=cutoff
                assert reply['status']==206 and reply['etag']==h['etag']
                assert reply['content_range']==f"bytes {o['start']}-{o['end']}/{h['content_length']}"
                # obj.end is the byte end after selection, interval.end is the forecast-hour end.
                b=(root/o['path']).read_bytes();paths.add(o['path']);payloads+=1;bytes_+=len(b)
                assert hashlib.sha256(b).hexdigest()==o['sha256']
                assert len(b)==o['end']-o['start']+1 and struct.unpack('>Q',b[8:16])[0]==len(b)
                assert b[:4]==b'GRIB' and b[7]==2 and b[-4:]==b'7777'
                s1=struct.unpack('>IBHHBBBH7B',b[16:37]);assert s1[1]==1 and s1[6]==1 and s1[-2:]==(0,4)
                assert datetime(*s1[7:13],tzinfo=timezone.utc)==init
                assert o['index_record']['offset']==o['start'] and o['index_record']['variable']=='APCP'
                nxt=o['next_index_record']
                assert o['end']==(nxt['offset']-1 if nxt else int(h['content_length'])-1)
                cursor=16;sections={}
                while cursor<len(b)-4:
                    length=struct.unpack('>I',b[cursor:cursor+4])[0];assert length>=5
                    number=b[cursor+4];sections[number]=b[cursor:cursor+length];cursor+=length
                assert cursor==len(b)-4
                s4=sections[4];assert struct.unpack('>H',s4[7:9])[0]==11
                assert s4[9:11]==bytes([1,8]) and s4[17]==1 and s4[22]==1
                assert int.from_bytes(s4[18:22],'big')==interval['end']-6
                assert s4[35]==member and s4[44]==1 and s4[49]==1 and s4[51]==1
                assert int.from_bytes(s4[52:56],'big')==6
                valid=datetime(int.from_bytes(s4[37:39],'big'),*s4[39:44],tzinfo=timezone.utc)
                assert valid==init+timedelta(hours=interval['end'])
                handle=E.codes_new_from_message(b)
                try:
                    m=o['metadata'];assert all(E.codes_get(handle,k)==v for k,v in m.items())
                    assert m['perturbationNumber']==member and m['numberOfForecastsInEnsemble']==s4[36]
                    ni,nj=m['Ni'],m['Nj']
                    assert E.codes_get(handle,'jPointsAreConsecutive')==0 and E.codes_get(handle,'iScansNegatively')==0 and E.codes_get(handle,'jScansPositively')==0
                    la0,lo0,di,dj=(E.codes_get(handle,k) for k in ('latitudeOfFirstGridPointInDegrees','longitudeOfFirstGridPointInDegrees','iDirectionIncrementInDegrees','jDirectionIncrementInDegrees'))
                    identity=(ni,nj,la0,lo0,di,dj,p['lat'],p['lon'])
                    if identity not in coordinates:
                        la=np.repeat(la0-dj*np.arange(nj),ni);lo=np.tile(lo0+di*np.arange(ni),nj);rad=math.pi/180
                        a=np.sin((la-p['lat'])*rad/2)**2+np.cos(la*rad)*math.cos(p['lat']*rad)*np.sin((lo-p['lon'])*rad/2)**2
                        ix=int(np.argmin(a));distance=2*6371.229*math.asin(math.sqrt(a[ix]))
                        coordinates[identity]=(ix,distance,float(la[ix]),float(lo[ix]))
                    ix,distance,lat,lon=coordinates[identity];point=o['nearest']
                    assert ix==point['index'] and lat==point['lat'] and lon==point['lon'] and abs(distance-point['distance'])<.01
                    value=float(E.codes_get_array(handle,'values')[ix]);assert value==point['value']==o['value']
                    total+=value*interval['overlap_hours']/6
                finally:E.codes_release(handle)
            assert abs(total-r['forecast_rain'])<1e-12
    with (root/'results/e226_ensemble_windows.csv').open() as f:windows=list(csv.DictReader(f))
    assert len(windows)==82 and len(groups)==82
    coverage={str(y):dict(total=0,complete=0) for y in range(2019,2023)}
    for row in windows:
        key=int(row['season']),int(row['round']);members=groups[key]
        assert sorted(members)==list(range(1,21))
        complete=all(r['complete'] for r in members.values())
        base=float(control[key]['forecast_precip_mm']>1)
        expected=(int(base)+sum(r['forecast_rain']>1 for r in members.values()))/21 if complete else base
        assert float(row['p_wet'])==expected and (row['complete']=='True')==complete
        coverage[str(key[0])]['total']+=1;coverage[str(key[0])]['complete']+=int(complete)
    result=json.loads((root/'results/e226_source_summary.json').read_text())
    gate=sum(v['complete'] for v in coverage.values())>=78 and all(v['complete']/v['total']>=.8 for v in coverage.values())
    assert coverage==result['coverage'] and gate==result['source_gate']
    print(json.dumps(dict(member_tasks=1640,complete_members=sum(r['complete'] for r in receipts),complete_payloads=payloads,bytes=bytes_,source_gate=gate),indent=2))


if __name__=='__main__':test_saved_source()
