"""Independent numeric-only artifact schema/count/range/gate consistency."""
import csv,gzip,json,math,re
from collections import defaultdict
from f1.data import ROOT


def test_artifact():
    source=json.loads((ROOT/'results/e230_native_schema.json').read_text())
    assert len(source['sources'])==6
    roster=defaultdict(set)
    with (ROOT/'results/e217_driver_numbers.csv').open() as f:
        for r in csv.DictReader(f):roster[(int(r['season']),int(r['round']))].add(r['RacingNumber'])
    qualified=defaultdict(int);total_records=total_values=total_bytes=0
    for r in source['sources']:
        key=r['season'],r['round'];feed=r['feed']
        assert key in {(2019,5),(2019,12),(2019,14)} and feed in {'DriverScore','SPFeed'}
        assert r['available'] and r['status']==200 and r['requests']==[dict(url=r['url'],status=200)]
        assert r['url'].endswith('_Race/'+feed+'.jsonStream') and '/static/2019/' in r['url']
        assert 0<=r['raw_bytes']<=4*1024*1024 and re.fullmatch('[0-9a-f]{64}',r['raw_sha256'])
        records=json.loads(gzip.decompress((ROOT/f'results/e230_2019_{r["round"]}_{feed}_numeric.json.gz').read_bytes()))
        fields={};cars=set();values=0
        for record in records:
            assert set(record)=={'seconds','numeric','types','candidate_car_keys'}
            assert 0<=record['seconds']<360000
            assert set(record['candidate_car_keys'])<=roster[key]
            cars.update(record['candidate_car_keys'])
            assert record['types'].get('number',0)+record['types'].get('numeric_string',0)==len(record['numeric'])
            for item in record['numeric']:
                assert set(item)=={'path','value'} and isinstance(item['value'],(int,float)) and math.isfinite(item['value'])
                path=item['path'];assert all(part=='[]' or part=='<other-key>' or re.fullmatch('[A-Za-z0-9_.-]{1,64}',part) for part in path)
                assert all(part.lower() not in {'message','text','msg','comment','commentary','description','audio','path','status','recording_url','recordingurl'} for part in path)
                canonical=tuple('*' if re.fullmatch('[0-9]+',part) else part for part in path)
                count,low,high=fields.get(canonical,(0,float('inf'),float('-inf')))
                fields[canonical]=(count+1,min(low,item['value']),max(high,item['value']));values+=1
        expected=[]
        for path,(count,low,high) in sorted(fields.items()):
            expected.append(dict(path=list(path),count=count,minimum=low,maximum=high,
                candidate_measurement=bool(path and path[-1].lower() not in {'racingnumber','drivernumber','lap','lapnumber','position','line','utc','time','timestamp'} and count>=10 and high>low)))
        assert expected==r['profiles'] and len(records)==r['records'] and values==r['numeric_leaves']
        assert sorted(cars)==r['candidate_car_keys']
        good=bool(records and r['malformed_records']/(len(records)+r['malformed_records'])<=.05 and len(cars)>=10 and any(p['candidate_measurement'] for p in expected))
        assert good==r['qualified'];qualified[feed]+=int(good)
        total_records+=len(records);total_values+=values;total_bytes+=r['raw_bytes']
    assert dict(qualified)==source['qualified_sessions']=={'DriverScore':3,'SPFeed':3}
    assert source['gate']==any(n>=2 for n in qualified.values()) and not source['winner_fit']
    print('Independent numeric-only profiles/gate/scope audit passes:',total_records,'records,',total_values,'numeric leaves,',total_bytes,'source bytes')
    print('Raw bodies not retained; array positions not preserved; semantic identities remain unverified')


if __name__=='__main__':test_artifact()
