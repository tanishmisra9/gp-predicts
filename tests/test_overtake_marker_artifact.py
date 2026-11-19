"""E239 independent positional histories via indexed look-back, not rolling state."""
import bisect
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import fastf1
import f1.ff1
from experiments.e190_traffic_feasibility import cached_session


def test_artifacts():
    result=json.loads(Path('results/e239_marker_validation.json').read_text())
    source=json.loads(Path('results/e238_driver_race_schema.json').read_text())
    assert [(r['season'],r['round']) for r in result['sources']]==[(2019,5),(2019,12),(2019,14)]
    fastf1.Cache.offline_mode(True); passes=0
    for report in result['sources']:
        session=cached_session(fastf1,2019,report['round'])
        roster=set(session.laps.DriverNumber.dropna().astype(str))
        raw=Path(f"results/e238_2019_{report['round']}_numeric.json.gz").read_bytes()
        assert report['projection_sha256']==hashlib.sha256(raw).hexdigest()
        receipt=next(r for r in source['sources'] if r['round']==report['round'])
        assert report['raw_source_sha256']==receipt['raw_sha256']
        records=json.loads(gzip.decompress(raw)); timelines=defaultdict(list); patches=[]; states=[]
        for index,record in enumerate(records):
            positions={}; flags={}
            for leaf in record['numeric']:
                path=leaf['path']
                if len(path)!=2 or path[1] not in ('Position','OvertakeState'): continue
                car,field=path; value=leaf['value']
                assert car in roster and value==int(value)
                if field=='Position':
                    assert car not in positions
                    positions[car]=int(value)
                    timelines[car].append((index,int(value) if 1<=value<=len(roster) else None))
                else:
                    assert car not in flags and int(value) in (0,1,2)
                    flags[car]=int(value)
            patches.append(positions); states.append(flags)
        keys={car:[p[0] for p in timeline] for car,timeline in timelines.items()}
        def prior(car,index):
            where=bisect.bisect_left(keys.get(car,[]),index)-1
            return timelines[car][where][1] if where>=0 else None
        expected=[]
        for index,flags in enumerate(states):
            for actor,state in flags.items():
                if state!=2: continue
                before=prior(actor,index); after=patches[index].get(actor)
                gained=bool(before is not None and after is not None and 1<=after<=len(roster) and before>after)
                others=[]
                if gained:
                    for other,new in patches[index].items():
                        old=prior(other,index)
                        if other!=actor and flags.get(other)!=2 and old is not None and 1<=new<=len(roster) and new>old and before>old and after<new:
                            others.append(other)
                expected.append(dict(record=index,seconds=records[index]['seconds'],car=actor,before=before,after=after,
                    known_prior=before is not None,actor_gain=gained,counterparts=sorted(others),echo=bool(others)))
        order=lambda x:(x['record'],x['car'])
        assert sorted(report['markers'],key=order)==sorted(expected,key=order)
        known=[r for r in expected if r['known_prior']]
        echoes=sum(r['echo'] for r in known)
        coverage=len(known)/len(expected) if expected else 0.
        fraction=echoes/len(known) if known else 0.
        qualified=bool(len(known)>=50 and coverage>=.95 and fraction>=.9)
        metrics=dict(state2_markers=len(expected),known_prior_markers=len(known),prior_coverage=coverage,
            actor_gains=sum(r['actor_gain'] for r in known),echo_markers=echoes,echo_fraction=fraction,qualified=qualified)
        assert all(report[k]==v for k,v in metrics.items())
        passes+=qualified
        print(report['round'],metrics)
    assert result['gate']==(passes>=2) and not result['winner_fit']
    assert not result['catching_meaning_verified'] and not result['ontrack_passes_verified']


if __name__=='__main__':
    test_artifacts()
    print('E239 independent indexed prior positions/all state-2 markers/reversals/counts/gates PASS')
