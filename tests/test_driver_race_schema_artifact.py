"""Independently audit E238 saved numeric schema and exact physical-field gate."""
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import fastf1
import f1.ff1
from experiments.e190_traffic_feasibility import cached_session

CANDIDATES=set('fuel fuelload fuellevel fuelremaining fuelconsumption energy ersenergy erscharge batterycharge stateofcharge enginetemperature watertemperature oiltemperature tyretemperature tiretemperature tyrepressure tirepressure steering steeringangle brakepressure gforce longitudinalg lateralg lateralacceleration longitudinalacceleration'.split())
EXCLUDE=set('racingnumber drivernumber lap lapnumber position line utc time timestamp'.split())


def test_artifact():
    result=json.loads(Path('results/e238_driver_race_schema.json').read_text())
    assert [(r['season'],r['round']) for r in result['sources']]==[(2019,5),(2019,12),(2019,14)]
    fastf1.Cache.offline_mode(True); passes=0; total=0
    for report in result['sources']:
        session=cached_session(fastf1,2019,report['round'])
        roster=set(session.laps.DriverNumber.dropna().astype(str))
        expected_url='https://livetiming.formula1.com'+session.api_path+'DriverRaceInfo.jsonStream'
        assert report['url']==expected_url
        assert 1<=len(report['requests'])<=2 and all(r['url']==expected_url for r in report['requests'])
        assert report['available'] and report['status']==200 and report['requests'][-1]['status']==200
        assert re.fullmatch('[0-9a-f]{64}',report['raw_sha256']) and 0<report['raw_bytes']<=4*1024*1024
        path=Path(f"results/e238_2019_{report['round']}_numeric.json.gz")
        assert int.from_bytes(path.read_bytes()[4:8],'little')==0
        records=json.loads(gzip.decompress(path.read_bytes()))
        profiles=defaultdict(list); car_keys=set(); physical_cars=set(); physical=[]; physical_paths=set(); leaves=0
        for record in records:
            assert set(record)=={'seconds','numeric','types','candidate_car_keys'}
            assert math.isfinite(record['seconds']) and set(record['candidate_car_keys'])<=roster
            car_keys.update(record['candidate_car_keys'])
            for leaf in record['numeric']:
                assert set(leaf)=={'path','value'} and isinstance(leaf['value'],float) and math.isfinite(leaf['value'])
                raw=leaf['path']; assert all(isinstance(p,str) for p in raw)
                path=tuple('*' if p.isdigit() else p for p in raw)
                profiles[path].append(leaf['value']); leaves+=1
                cleaned=[''.join(c for c in part.casefold() if c in 'abcdefghijklmnopqrstuvwxyz0123456789') for part in raw]
                if set(cleaned)&CANDIDATES:
                    physical.append(leaf['value']); physical_cars.update(set(raw)&roster); physical_paths.add(path)
        expected=[]
        for path,values in sorted(profiles.items()):
            expected.append(dict(path=list(path),count=len(values),minimum=min(values),maximum=max(values),
                candidate_measurement=bool(path and path[-1].lower() not in EXCLUDE and len(values)>=10 and max(values)>min(values))))
        assert report['profiles']==expected and report['numeric_leaves']==leaves and report['records']==len(records)
        assert report['candidate_car_keys']==sorted(car_keys)
        assert report['physical_leaves']==len(physical) and report['physical_car_keys']==sorted(physical_cars)
        assert report['physical_paths']==[list(p) for p in sorted(physical_paths)]
        quality=bool(records and report['malformed_records']/(len(records)+report['malformed_records'])<=.05)
        qualified=bool(quality and len(physical)>=100 and len(physical_cars)>=10 and len(set(physical))>=2)
        assert report['physical_qualified']==qualified
        assert report['qualified']==bool(quality and len(car_keys)>=10 and any(p['candidate_measurement'] for p in expected))
        passes+=qualified;total+=leaves
        print(report['round'],'records',len(records),'numeric',leaves,'physical',len(physical),'gate',qualified)
    assert result['gate']==(passes>=2) and not result['winner_fit'] and not result['physical_meanings_verified']
    print('all numeric leaves independently counted:',total)


if __name__=='__main__':
    test_artifact()
    print('E238 independent projection/scope/physical-mask/count/range/gate audit PASS; raw bodies not retained')
