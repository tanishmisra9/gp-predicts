"""E242 bounded historical constructor-standing source and point-sum audit."""
import gzip
import hashlib
import json
import math
import re
import time
from unittest.mock import patch

import requests

from experiments.e196_weather_source import restricted_send
from f1.data import BASE,ROOT,_s,load_results

PILOT=((2020,17,'2021-01-01'),(2021,8,'2021-07-01'),(2021,22,'2022-01-01'))
RESULT=ROOT/'results/e242_constructor_snapshots.json'


def project(payload,year,round_):
    data=payload['MRData'];table=data['StandingsTable'];lists=table['StandingsLists']
    if int(table['season'])!=year or int(table['round'])!=round_ or len(lists)!=1:
        raise ValueError('snapshot identity mismatch')
    standings=lists[0]
    if int(standings['season'])!=year or int(standings['round'])!=round_ or int(data['offset'])!=0 or int(data['total'])!=10:
        raise ValueError('snapshot identity/pagination mismatch')
    rows=[]
    for raw in standings['ConstructorStandings']:
        constructor=raw['Constructor']['constructorId']
        if not isinstance(constructor,str) or not re.fullmatch('[a-z0-9_]{1,64}',constructor):
            raise ValueError('invalid constructor identity')
        values={}
        for name in ('position','points','wins'):
            value=raw[name]
            if isinstance(value,bool):raise ValueError('boolean standing number')
            number=float(value)
            if not math.isfinite(number) or number<0 or (name!='points' and not number.is_integer()):
                raise ValueError('invalid standing number')
            values[name]=number if name=='points' else int(number)
        if values['wins']>round_:raise ValueError('wins exceed snapshot round')
        rows.append(dict(constructor=constructor,**values))
    if len(rows)!=10 or len({r['constructor'] for r in rows})!=10 or {r['position'] for r in rows}!=set(range(1,11)):
        raise ValueError('incomplete/duplicate constructor standings')
    return sorted(rows,key=lambda r:r['position'])


def compare(rows,history):
    assert history.season.nunique()==1 and not history.duplicated(['season','round','driver']).any()
    race=history.groupby('team').points.sum()
    sprint=history.assign(sprint=history.s_points.fillna(0)).groupby('team').sprint.sum()
    assert set(race.index)=={r['constructor'] for r in rows}
    compared=[]
    for row in rows:
        c=row['constructor'];r=float(race[c]);combined=r+float(sprint[c])
        compared.append(dict(**row,race_points=r,race_plus_sprint_points=combined,
            official_minus_race=row['points']-r,official_minus_combined=row['points']-combined))
    inversions=[]
    for i,better in enumerate(compared):
        for worse in compared[i+1:]:
            if better['race_plus_sprint_points']<worse['race_plus_sprint_points']:
                inversions.append([better['constructor'],worse['constructor']])
    return compared,inversions


def main():
    import pandas as pd
    df=load_results().loc[lambda d:d.season.between(2020,2021)]
    reports=json.loads(RESULT.read_text())['sources'] if RESULT.exists() else []
    done={(r['season'],r['round']) for r in reports}
    assert len(done)==len(reports) and done.issubset({(y,r) for y,r,_ in PILOT})
    for year,round_,reference in PILOT:
        if (year,round_) in done:continue
        row=dict(season=year,round=round_,reference_date=reference,available=False,qualified=False,requests=[])
        url=f'{BASE}/{year}/{round_}/constructorstandings/'
        params={'limit':100,'offset':0}
        exact=requests.Request('GET',url,params=params).prepare().url
        row['url']=exact
        try:
            with patch('requests.adapters.HTTPAdapter.send',new=restricted_send(requests.adapters.HTTPAdapter.send,{exact},row['requests'])):
                response=_s.get(url,params=params,headers={'Accept':'application/json'},timeout=20,allow_redirects=False)
            row.update(status=int(response.status_code),from_cache=bool(getattr(response,'from_cache',False)),http_date=response.headers.get('Date'))
            if not row['from_cache']:time.sleep(.4)
            if response.status_code!=200:raise ValueError('snapshot unavailable')
            raw=response.content
            if len(raw)>1024*1024:raise ValueError('snapshot exceeds acceptance bound')
            standings=project(json.loads(raw),year,round_)
            history=df[df.season.eq(year)&df['round'].le(round_)]
            date=history.loc[history['round'].eq(round_),'date'].drop_duplicates()
            assert len(date)==1
            reference_time=pd.Timestamp(reference)
            if not date.iloc[0]<reference_time:raise ValueError('reference precedes source event')
            compared,inversions=compare(standings,history)
            (ROOT/f'results/e242_{year}_{round_}_raw.json.gz').write_bytes(gzip.compress(raw,mtime=0))
            row.update(available=True,qualified=True,raw_bytes=len(raw),raw_sha256=hashlib.sha256(raw).hexdigest(),
                event_date=date.iloc[0].strftime('%Y-%m-%d'),constructors=compared,combined_point_inversions=inversions)
        except Exception as exc:
            row['error_type']=type(exc).__name__
        reports.append(row)
        result=dict(sources=reports,complete=len(reports)==len(PILOT),gate=bool(len(reports)==len(PILOT) and sum(r['qualified'] for r in reports)>=2),
            resource_exposure_computed=False,entitlement_verified=False,winner_fit=False,historical_publication_verified=False)
        RESULT.write_text(json.dumps(result,indent=2)+'\n')
        print('SOURCE',year,round_,row.get('status'),row.get('from_cache'),row['qualified'],flush=True)


if __name__=='__main__':
    main()
