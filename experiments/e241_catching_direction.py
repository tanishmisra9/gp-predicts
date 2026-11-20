"""E241 emitted Catching-code association with recent same-opponent Interval sign."""
import gzip
import hashlib
import json
from collections import Counter

from f1.data import ROOT


def analyse(records,roster):
    positions={}; intervals={}; events=[]
    def ahead(car):
        position=positions.get(car)
        if position is None or position<=1: return None
        candidates=[d for d,p in positions.items() if p==position-1]
        return candidates[0] if len(candidates)==1 else None
    for index,record in enumerate(records):
        patch={}
        for leaf in record['numeric']:
            path=leaf['path']
            if len(path)!=2 or path[1] not in ('Position','Interval','Catching'): continue
            car,field=path
            if car not in roster: raise ValueError('unknown driver')
            fields=patch.setdefault(car,{})
            if field in fields: raise ValueError('duplicate coordinate')
            fields[field]=float(leaf['value'])
            if field=='Catching' and fields[field] not in (1,2): raise ValueError('unregistered Catching code')
        for car,fields in patch.items():
            if 'Position' in fields:
                value=fields['Position']
                if value.is_integer() and 1<=value<=len(roster): positions[car]=int(value)
                else: positions.pop(car,None)
        for car,fields in patch.items():
            now=fields.get('Interval'); previous=intervals.get(car)
            position=positions.get(car); opponent=ahead(car)
            if 'Catching' in fields:
                event=dict(record=index,seconds=record['seconds'],car=car,code=int(fields['Catching']),position=position,ahead=opponent,
                    current=now,scored=False,reason='missing_pair')
                if now is not None and now>=0 and previous is not None:
                    old,time,old_position,old_ahead=previous
                    age=record['seconds']-time
                    event.update(previous=old,age=age)
                    if not 0<=age<=10: event['reason']='stale_or_clock'
                    elif position is None or opponent is None or position!=old_position or opponent!=old_ahead: event['reason']='changed_or_unknown_opponent'
                    elif now==old: event['reason']='no_change'
                    else: event.update(scored=True,reason='scored',delta=now-old,decreasing=now<old)
                events.append(event)
            if now is not None:
                if now>=0: intervals[car]=(now,record['seconds'],position,opponent)
                else: intervals.pop(car,None)
    scored=[e for e in events if e['scored']]
    table=Counter((e['code'],'decrease' if e['decreasing'] else 'increase') for e in scored)
    n=len(scored); a=table[(1,'decrease')]+table[(2,'increase')]
    support=bool(n>=100 and min(sum(e['code']==c for e in scored) for c in (1,2))>=50)
    return dict(coded_updates=len(events),scored_updates=n,reasons=dict(Counter(e['reason'] for e in events)),
        contingency={f'{code}_{direction}':table[(code,direction)] for code in (1,2) for direction in ('decrease','increase')},
        support_gate=support,mapping_a_accuracy=a/n if n else None,mapping_b_accuracy=(n-a)/n if n else None,
        favoured='A' if a>n-a else ('B' if a<n-a else None),
        qualified_a=bool(support and a/n>=.95),qualified_b=bool(support and (n-a)/n>=.95),events=events)


def main():
    manifest=json.loads((ROOT/'results/e238_driver_race_schema.json').read_text()); sources=[]
    for round_ in (5,12,14):
        receipt=next(r for r in manifest['sources'] if r['round']==round_)
        path=ROOT/f'results/e238_2019_{round_}_numeric.json.gz'; raw=path.read_bytes()
        summary=analyse(json.loads(gzip.decompress(raw)),set(receipt['candidate_car_keys']))
        sources.append(dict(season=2019,round=round_,projection_sha256=hashlib.sha256(raw).hexdigest(),**summary))
        print('SOURCE',round_,json.dumps({k:v for k,v in summary.items() if k!='events'}),flush=True)
    favoured={s['favoured'] for s in sources}
    consistent=len(favoured)==1 and None not in favoured
    a=sum(s['qualified_a'] for s in sources); b=sum(s['qualified_b'] for s in sources)
    gate=bool(consistent and max(a,b)>=2)
    result=dict(sources=sources,consistent=consistent,qualified_a=a,qualified_b=b,gate=gate,
        mapping=('A' if a>=2 else 'B') if gate else None,winner_fit=False,official_formula_verified=False)
    (ROOT/'results/e241_catching_direction.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
