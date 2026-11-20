"""Independent E242 raw public JSON projection and scalar point-sum audit."""
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from f1.data import load_results


def test_artifacts():
    result=json.loads(Path('results/e242_constructor_snapshots.json').read_text())
    expected=[(2020,17,'2021-01-01'),(2021,8,'2021-07-01'),(2021,22,'2022-01-01')]
    assert [(r['season'],r['round'],r['reference_date']) for r in result['sources']]==expected
    history=load_results().loc[lambda d:d.season.between(2020,2021)]
    good=0
    for report in result['sources']:
        year,round_=report['season'],report['round']
        url=f'https://api.jolpi.ca/ergast/f1/{year}/{round_}/constructorstandings/?limit=100&offset=0'
        assert report['url']==url and len(report['requests'])<=1
        assert all(r['url']==url and r['status']==200 for r in report['requests'])
        assert report['qualified'] and report['available'] and report['status']==200
        assert len(report['requests'])==(0 if report['from_cache'] else 1)
        packed=Path(f'results/e242_{year}_{round_}_raw.json.gz').read_bytes()
        assert int.from_bytes(packed[4:8],'little')==0
        raw=gzip.decompress(packed)
        assert report['raw_bytes']==len(raw)<=1024*1024 and report['raw_sha256']==hashlib.sha256(raw).hexdigest()
        payload=json.loads(raw)['MRData'];table=payload['StandingsTable']
        assert int(payload['total'])==10 and int(payload['offset'])==0
        assert (int(table['season']),int(table['round']))==(year,round_) and len(table['StandingsLists'])==1
        entry=table['StandingsLists'][0]
        assert (int(entry['season']),int(entry['round']))==(year,round_)
        rows=[]
        for row in entry['ConstructorStandings']:
            constructor=row['Constructor']['constructorId']
            assert re.fullmatch('[a-z0-9_]{1,64}',constructor)
            position=int(row['position']);points=float(row['points']);wins=int(row['wins'])
            assert math.isfinite(points) and points>=0 and 0<=wins<=round_
            rows.append(dict(constructor=constructor,position=position,points=points,wins=wins))
        rows.sort(key=lambda r:r['position'])
        assert len(rows)==len({r['constructor'] for r in rows})==10 and [r['position'] for r in rows]==list(range(1,11))
        data=history[history.season.eq(year)&history['round'].le(round_)]
        dates=set(data.loc[data['round'].eq(round_),'date'])
        assert len(dates)==1 and next(iter(dates)).strftime('%Y-%m-%d')==report['event_date']
        assert next(iter(dates))<pd.Timestamp(report['reference_date'])
        race=defaultdict(list);sprint=defaultdict(list)
        for r in data.itertuples():
            race[r.team].append(float(r.points) if pd.notna(r.points) else 0.)
            sprint[r.team].append(float(r.s_points) if pd.notna(r.s_points) else 0.)
        assert set(race)=={r['constructor'] for r in rows}
        compared=[]
        for row in rows:
            constructor=row['constructor'];r=math.fsum(race[constructor]);both=r+math.fsum(sprint[constructor])
            compared.append(dict(**row,race_points=r,race_plus_sprint_points=both,
                official_minus_race=row['points']-r,official_minus_combined=row['points']-both))
        assert report['constructors']==compared
        inversions=[]
        for i,a in enumerate(compared):
            for b in compared[i+1:]:
                if a['race_plus_sprint_points']<b['race_plus_sprint_points']:
                    inversions.append([a['constructor'],b['constructor']])
        assert report['combined_point_inversions']==inversions
        good+=1
        print(year,round_,'constructors',len(compared),'combined-point order inversions',inversions)
    assert result['complete'] and result['gate']==(good>=2)
    for flag in ('resource_exposure_computed','entitlement_verified','winner_fit','historical_publication_verified'):
        assert not result[flag]


if __name__=='__main__':
    test_artifacts()
    print('E242 independent raw-body/schema/identity/hash/calendar/prefix-points/inversion audit PASS')
