"""E241 independent indexed Interval and transactional position-context lookbacks."""
import bisect
import gzip
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path


def test_artifact():
    result=json.loads(Path('results/e241_catching_direction.json').read_text())
    manifest=json.loads(Path('results/e238_driver_race_schema.json').read_text())
    assert [(s['season'],s['round']) for s in result['sources']]==[(2019,5),(2019,12),(2019,14)]
    a_count=0;b_count=0;favoured=set()
    for report in result['sources']:
        receipt=next(s for s in manifest['sources'] if s['round']==report['round'])
        roster=set(receipt['candidate_car_keys'])
        raw=Path(f"results/e238_2019_{report['round']}_numeric.json.gz").read_bytes()
        assert report['projection_sha256']==hashlib.sha256(raw).hexdigest()
        records=json.loads(gzip.decompress(raw))
        positions=defaultdict(list);intervals=defaultdict(list);patches=[]
        for index,record in enumerate(records):
            patch={}
            for leaf in record['numeric']:
                path=leaf['path']
                if len(path)!=2 or path[1] not in ('Position','Interval','Catching'): continue
                car,field=path;value=leaf['value']
                assert car in roster and (car,field) not in patch
                patch[(car,field)]=value
                if field=='Position': positions[car].append((index,int(value) if value==int(value) and 1<=value<=len(roster) else None))
                elif field=='Interval': intervals[car].append((index,value if value>=0 else None))
                else: assert value in (1,2)
            patches.append(patch)
        position_keys={c:[v[0] for v in values] for c,values in positions.items()}
        interval_keys={c:[v[0] for v in values] for c,values in intervals.items()}
        def position(car,index):
            where=bisect.bisect_right(position_keys.get(car,[]),index)-1
            return positions[car][where][1] if where>=0 else None
        def context(car,index):
            own=position(car,index)
            if own is None or own<=1:return own,None
            leaders=[c for c in roster if position(c,index)==own-1]
            return own,leaders[0] if len(leaders)==1 else None
        expected=[]
        for index,patch in enumerate(patches):
            for (car,field),value in patch.items():
                if field!='Catching': continue
                own,opponent=context(car,index);current=patch.get((car,'Interval'))
                event=dict(record=index,seconds=records[index]['seconds'],car=car,code=int(value),position=own,ahead=opponent,
                    current=current,scored=False,reason='missing_pair')
                prior=bisect.bisect_left(interval_keys.get(car,[]),index)-1
                if current is not None and current>=0 and prior>=0 and intervals[car][prior][1] is not None:
                    old_index,old=intervals[car][prior]
                    age=records[index]['seconds']-records[old_index]['seconds']
                    old_own,old_opponent=context(car,old_index)
                    event.update(previous=old,age=age)
                    if age<0 or age>10:event['reason']='stale_or_clock'
                    elif own is None or opponent is None or own!=old_own or opponent!=old_opponent:event['reason']='changed_or_unknown_opponent'
                    elif current==old:event['reason']='no_change'
                    else:event.update(scored=True,reason='scored',delta=current-old,decreasing=current<old)
                expected.append(event)
        key=lambda e:(e['record'],e['car'])
        assert sorted(report['events'],key=key)==sorted(expected,key=key)
        scored=[e for e in expected if e['scored']]
        counts=Counter((e['code'],'decrease' if e['decreasing'] else 'increase') for e in scored)
        n=len(scored);a=counts[(1,'decrease')]+counts[(2,'increase')]
        support=n>=100 and all(sum(e['code']==c for e in scored)>=50 for c in (1,2))
        metrics=dict(coded_updates=len(expected),scored_updates=n,reasons=dict(Counter(e['reason'] for e in expected)),
            contingency={f'{c}_{d}':counts[(c,d)] for c in (1,2) for d in ('decrease','increase')},support_gate=support,
            mapping_a_accuracy=a/n if n else None,mapping_b_accuracy=(n-a)/n if n else None,
            favoured='A' if a>n-a else ('B' if a<n-a else None),qualified_a=bool(support and a/n>=.95),qualified_b=bool(support and (n-a)/n>=.95))
        assert all(report[k]==v for k,v in metrics.items())
        a_count+=metrics['qualified_a'];b_count+=metrics['qualified_b'];favoured.add(metrics['favoured'])
        print(report['round'],'coded',len(expected),'paired',n,'A/B',metrics['mapping_a_accuracy'],metrics['mapping_b_accuracy'])
    consistent=len(favoured)==1 and None not in favoured
    gate=bool(consistent and max(a_count,b_count)>=2)
    assert result['consistent']==consistent and result['qualified_a']==a_count and result['qualified_b']==b_count and result['gate']==gate
    assert result['mapping']==(('A' if a_count>=2 else 'B') if gate else None)
    assert not result['winner_fit'] and not result['official_formula_verified']


if __name__=='__main__':
    test_artifact()
    print('E241 independent indexed positions/Interval priors/opponents/events/cells/mappings/gates PASS')
