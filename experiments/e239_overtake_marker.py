"""E239 fixed same-patch position reversal validation of native state 2."""
import gzip
import hashlib
import json

from experiments.e190_traffic_feasibility import PILOT,cached_session
from f1.data import ROOT


def analyse(records,roster):
    positions={}; markers=[]
    for index,record in enumerate(records):
        patch={}
        for leaf in record['numeric']:
            path=leaf['path']
            if len(path)!=2 or path[1] not in ('Position','OvertakeState'):
                continue
            car,field=path
            if car not in roster:
                raise ValueError('unknown driver key')
            fields=patch.setdefault(car,{})
            if field in fields:
                raise ValueError('duplicate numeric coordinate')
            value=leaf['value']
            if not float(value).is_integer():
                raise ValueError('noninteger position/state')
            fields[field]=int(value)
            if field=='OvertakeState' and int(value) not in (0,1,2):
                raise ValueError('unregistered state code')
        for car,fields in patch.items():
            if fields.get('OvertakeState')!=2:
                continue
            before=positions.get(car); after=fields.get('Position')
            valid_after=after is not None and 1<=after<=len(roster)
            gained=bool(before is not None and valid_after and after<before)
            counterparts=[]
            if gained:
                for other,p in patch.items():
                    old=positions.get(other); new=p.get('Position')
                    if other==car or p.get('OvertakeState')==2 or old is None or new is None or not 1<=new<=len(roster):
                        continue
                    if before>old and after<new and new>old:
                        counterparts.append(other)
            markers.append(dict(record=index,seconds=record['seconds'],car=car,before=before,after=after,
                known_prior=before is not None,actor_gain=gained,counterparts=sorted(counterparts),echo=bool(counterparts)))
        for car,fields in patch.items():
            if 'Position' in fields:
                value=fields['Position']
                if 1<=value<=len(roster): positions[car]=value
                else: positions.pop(car,None)
    known=[m for m in markers if m['known_prior']]
    echoed=sum(m['echo'] for m in known)
    coverage=len(known)/len(markers) if markers else 0.
    fraction=echoed/len(known) if known else 0.
    result=dict(state2_markers=len(markers),known_prior_markers=len(known),prior_coverage=coverage,
        actor_gains=sum(m['actor_gain'] for m in known),echo_markers=echoed,echo_fraction=fraction,
        qualified=bool(len(known)>=50 and coverage>=.95 and fraction>=.9))
    return result,markers


def main():
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True); reports=[]
    manifest=json.loads((ROOT/'results/e238_driver_race_schema.json').read_text())
    for season,round_ in PILOT:
        session=cached_session(fastf1,season,round_)
        roster=set(session.laps.DriverNumber.dropna().astype(str))
        path=ROOT/f'results/e238_2019_{round_}_numeric.json.gz'
        raw=path.read_bytes(); records=json.loads(gzip.decompress(raw))
        result,markers=analyse(records,roster)
        receipt=next(r for r in manifest['sources'] if r['round']==round_)
        reports.append(dict(season=season,round=round_,projection_sha256=hashlib.sha256(raw).hexdigest(),
            raw_source_sha256=receipt['raw_sha256'],**result,markers=markers))
        print('SOURCE',round_,json.dumps(result),flush=True)
    result=dict(sources=reports,gate=sum(r['qualified'] for r in reports)>=2,winner_fit=False,
        catching_meaning_verified=False,ontrack_passes_verified=False)
    (ROOT/'results/e239_marker_validation.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
