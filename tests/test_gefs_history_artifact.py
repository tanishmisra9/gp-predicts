"""Offline raw-source audit; run with the isolated weather environment."""

def test_saved_history():
    import csv,hashlib,json,math,struct
    from pathlib import Path
    from datetime import datetime,timedelta,timezone
    from unittest.mock import patch
    import numpy as np
    import eccodes as E
    root=Path.cwd(); rows=[json.loads(p.read_text()) for p in sorted((root/'results/e223_receipts').glob('*.json'))]
    assert len(rows)==103 and sum(r['complete'] for r in rows)==91
    expected={2018:(21,9),2019:(21,21),2020:(17,17),2021:(22,22),2022:(22,22)}
    assert {(y):(sum(r['plan']['season']==y for r in rows),sum(r['plan']['season']==y and r['complete'] for r in rows)) for y in expected}==expected
    summary=json.loads((root/'results/e223_gefs_coverage.json').read_text()); assert not summary['source_gate']
    count=size=0; used=set(); rain=[]; temps=[]
    with patch('requests.sessions.Session.request',side_effect=AssertionError('offline audit')):
     for r in rows:
      p=r['plan']; init=datetime.fromisoformat(p['initialization']); cutoff=datetime.fromisoformat(p['cutoff'])
      assert init+timedelta(hours=12)==cutoff and cutoff.weekday()==3 and init.weekday()==2
      assert sum(i['overlap_hours'] for i in p['intervals'])==6
      if not r['complete']:
       assert p['season']==2018 and p['round']<=12 and r['objects'][-1]['receipt']['status']==404
       assert 'forecast_temp_c' not in r and 'forecast_precip_mm' not in r
       continue
      t=a=0.
      for o in r['objects']:
       receipt=o['receipt']; modified=datetime.strptime(receipt['last_modified'],'%a, %d %b %Y %H:%M:%S GMT').replace(tzinfo=timezone.utc)
       assert init<modified<=cutoff and receipt['status']==200
       for var,f in o['fields'].items():
        path=root/f['path']; used.add(str(path)); b=path.read_bytes(); size+=len(b); count+=1
        assert len(b)==f['end']-f['start']+1 and hashlib.sha256(b).hexdigest()==f['sha256']
        assert b[:4]==b'GRIB' and b[7]==2 and struct.unpack('>Q',b[8:16])[0]==len(b) and b[-4:]==b'7777'
        sec=struct.unpack('>IBHHBBBH7B',b[16:37]); assert sec[1]==1 and sec[6]==1 and sec[-2:]==(0,3)
        assert datetime(*sec[7:13],tzinfo=timezone.utc)==init
        h=E.codes_new_from_message(b)
        try:
         assert E.codes_get(h,'startStep')==(o['end'] if var=='TMP' else o['end']-6)
         assert E.codes_get(h,'endStep')==o['end'] and E.codes_get(h,'stepUnits')==1
         assert all(E.codes_get(h,k)==v for k,v in f['metadata'].items())
         assert datetime.strptime(f['last_modified'],'%a, %d %b %Y %H:%M:%S GMT').replace(tzinfo=timezone.utc)<=cutoff
         ni,nj=(E.codes_get(h,k) for k in ('Ni','Nj'))
         assert E.codes_get(h,'jPointsAreConsecutive')==0 and E.codes_get(h,'iScansNegatively')==0 and E.codes_get(h,'jScansPositively')==0
         lat0,lon0,di,dj=(E.codes_get(h,k) for k in ('latitudeOfFirstGridPointInDegrees','longitudeOfFirstGridPointInDegrees','iDirectionIncrementInDegrees','jDirectionIncrementInDegrees'))
         la=np.repeat(lat0-dj*np.arange(nj),ni); lo=np.tile(lon0+di*np.arange(ni),nj); rad=np.pi/180
         z=np.sin((la-p['lat'])*rad/2)**2+np.cos(la*rad)*math.cos(p['lat']*rad)*np.sin((lo-p['lon'])*rad/2)**2
         ix=int(np.argmin(z)); values=E.codes_get_array(h,'values')
         assert ix==f['nearest']['index'] and values[ix]==f['nearest']['value']
         assert abs(2*6371.229*math.asin(math.sqrt(z[ix]))-f['nearest']['distance'])<.01
         value=float(values[ix])-(273.15 if var=='TMP' else 0)
         assert value==f['value']
         if var=='TMP': t+=value*o['overlap_hours']/6
         else:a+=value*o['overlap_hours']/6
        finally:E.codes_release(h)
      assert abs(t-r['forecast_temp_c'])<1e-12 and abs(a-r['forecast_precip_mm'])<1e-12
      temps.append(t);rain.append(a)
    assert count==352 and len(used)==352
    assert len(list((root/'data/parts/gefs_e223').glob('*.grib2')))==352
    print(json.dumps(dict(races=len(rows),complete=91,failed=12,messages=count,bytes=size,
                         temperature_range=[min(temps),max(temps)],precip_range=[min(rain),max(rain)],
                         positive_precip_races=sum(x>0 for x in rain),rain_gt1_races=sum(x>1 for x in rain)),indent=2))


if __name__=='__main__':
    test_saved_history()
