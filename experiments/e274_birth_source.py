"""E274 readonly reprojection of previously acquired official driver metadata."""
import csv
import datetime as dt
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch

import requests

from f1.data import BASE, ROOT, _s

CACHE = Path('/Users/tanishmisra/Code/gp-predicts/data/http_cache.sqlite')
TOTALS = {2018: 420, 2019: 420, 2020: 340, 2021: 440, 2022: 440}


def official_entries():
    with (ROOT / 'data/results.csv').open() as file:
        return [dict(season=int(r['season']), round=int(r['round']), driver=r['driver'], date=r['date'])
                for r in csv.DictReader(file) if 2018 <= int(r['season']) <= 2022]


def plan():
    return [dict(season=y, offset=offset, total=total,
                 url=f'{BASE}/{y}/results.json', params={'limit': 100, 'offset': offset})
            for y, total in TOTALS.items() for offset in range(0, total, 100)]


def page_metadata(body, item):
    data = json.loads(body)['MRData']
    assert int(data['limit']) == 100 and int(data['offset']) == item['offset'] and int(data['total']) == item['total']
    table = data['RaceTable']; assert int(table['season']) == item['season']
    assert isinstance(table['Races'], list)
    assert all(int(r['season']) == item['season'] for r in table['Races'])
    return table['Races']


def birth_date(value, event_date):
    if value is None or value == '':
        return None
    assert isinstance(value, str) and re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', value)
    date = dt.date.fromisoformat(value); event = dt.date.fromisoformat(event_date)
    assert date < event and 15 <= (event-date).days/365.2425 <= 60
    return value


def project(races, official):
    rows = []
    for race in races:
        year, rnd, date = int(race['season']), int(race['round']), race['date']
        assert year in TOTALS
        for result in race['Results']:
            driver = result['Driver']['driverId']; key = year, rnd, driver
            assert key in official and official[key] == date
            rows.append(dict(season=year, round=rnd, date=date, driver=driver,
                             dateOfBirth=birth_date(result['Driver'].get('dateOfBirth'), date)))
    assert len({(r['season'], r['round'], r['driver']) for r in rows}) == len(rows)
    return rows


def prior_records(rows, targets):
    output = []
    for target in targets:
        event = target['season'], target['round']
        date = dt.date.fromisoformat(target['date'])
        cutoff = date-dt.timedelta(days=(date.weekday()-3) % 7)
        past = [r for r in rows if r['driver'] == target['driver'] and r['dateOfBirth'] is not None
                and (r['season'], r['round']) < event and dt.date.fromisoformat(r['date']) < cutoff]
        dates = {r['dateOfBirth'] for r in past}
        output.append(dict(**target, prior_dob=next(iter(dates)) if len(dates) == 1 else None,
                           prior_conflict=len(dates) > 1))
    return output


def summarize(rows, official):
    keys = {(r['season'], r['round'], r['driver']) for r in official}
    assert len(keys) == len(official)
    observed = {(r['season'], r['round'], r['driver']): r for r in rows}
    assert len(observed) == len(rows) and set(observed) <= keys
    drivers = {r['driver'] for r in official}
    birth_sets = {driver: {r['dateOfBirth'] for r in rows if r['driver'] == driver and r['dateOfBirth'] is not None}
                  for driver in drivers}
    years = {}
    for year in TOTALS:
        entrants = [r for r in official if r['season'] == year]
        covered = sum(observed.get((r['season'], r['round'], r['driver']), {}).get('dateOfBirth') is not None for r in entrants)
        years[year] = dict(entries=len(entrants), covered=covered, coverage=covered/len(entrants))
    targets = [r for r in official if r['season'] >= 2020]
    prior = prior_records(rows, targets)
    prior_years = {year: dict(entries=sum(r['season'] == year for r in prior),
                             covered=sum(r['season'] == year and r['prior_dob'] is not None for r in prior))
                   for year in [2020, 2021, 2022]}
    for values in prior_years.values():
        values['coverage'] = values['covered']/values['entries']
    conflicts = {driver: sorted(dates) for driver, dates in sorted(birth_sets.items()) if len(dates) > 1}
    unique_covered = sum(len(dates) == 1 for dates in birth_sets.values())
    return dict(years=years, unique_drivers=len(drivers), unique_covered=unique_covered,
                official_entries=len(official), source_rows=len(rows), source_events=len({(r['season'], r['round']) for r in rows}),
                missing_entries=[r for r in official if (r['season'], r['round'], r['driver']) not in observed],
                missing_birth_dates=[r for r in rows if r['dateOfBirth'] is None],
                conflicts=conflicts, prior_years=prior_years,
                content_gate=bool(not conflicts and unique_covered/len(drivers) >= .95
                                  and all(v['coverage'] >= .95 for v in years.values())
                                  and all(v['coverage'] >= .95 for v in prior_years.values()))), prior


def main():
    target = ROOT / 'results/e274_birth_source_summary.json'
    assert not target.exists(), 'do not overwrite an audited source run'
    before = hashlib.sha256(CACHE.read_bytes()).hexdigest()
    official = official_entries()
    mapping = {(r['season'], r['round'], r['driver']): r['date'] for r in official}
    assert len(mapping) == len(official) == 2060
    reports = []; rows = []
    with sqlite3.connect(f'file:{CACHE}?mode=ro', uri=True) as connection, \
            patch('requests.adapters.HTTPAdapter.send', side_effect=AssertionError('E274 HTTP forbidden')) as sends:
        for item in plan():
            request = requests.Request('GET', item['url'], params=item['params']).prepare()
            key = _s.cache.create_key(request)
            report = dict(season=item['season'], offset=item['offset'], cache_key=key, available=False)
            found = connection.execute('SELECT value FROM responses WHERE key=?', (key,)).fetchone()
            if found is None:
                report['reason'] = 'not_cached'
            else:
                blob = found[0]
                response = _s.cache.responses.deserialize(key, blob)
                url = urlsplit(response.url)
                try:
                    assert response.status_code == 200 and response.from_cache
                    assert url.scheme == 'https' and url.hostname == 'api.jolpi.ca'
                    assert url.path in [f'/ergast/f1/{item["season"]}/results.json', f'/ergast/f1/{item["season"]}/results/']
                    assert parse_qs(url.query) == {'limit': ['100'], 'offset': [str(item['offset'])]}
                    races = page_metadata(response.content, item)
                except (AssertionError, ValueError, KeyError) as exc:
                    report['metadata_error_type'] = type(exc).__name__
                else:
                    report.update(available=True, url=response.url, status=200, body_bytes=len(response.content),
                                  body_sha256=hashlib.sha256(response.content).hexdigest(),
                                  blob_sha256=hashlib.sha256(blob).hexdigest())
                    rows.extend(project(races, mapping))
            reports.append(report)
        assert sends.call_count == 0
    assert hashlib.sha256(CACHE.read_bytes()).hexdigest() == before, 'original cache changed'
    summary, prior = summarize(rows, official)
    summary.update(pages=reports, source_gate=bool(len(reports) == 24 and all(r['available'] for r in reports) and summary['content_gate']),
                   original_cache_sha256=before, actual_http_sends=0, original_cache_unchanged=True,
                   historical_publication_time_verified=False, winner_fit=False, outer_evaluated=False, evaluated_2026=False,
                   artifact_hashes={})
    for name, value in [('birth_projection', rows), ('prior_birth_records', prior), ('birth_source_summary', summary)]:
        path = ROOT / f'results/e274_{name}.json'
        with path.open('x') as file:
            json.dump(value, file, indent=2); file.write('\n')
        if name != 'birth_source_summary':
            summary['artifact_hashes'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    print('SOURCE', json.dumps({k: v for k, v in summary.items() if k != 'pages'}), flush=True)


if __name__ == '__main__':
    main()
