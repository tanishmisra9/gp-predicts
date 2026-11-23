"""E274 independent readonly pages, full-entry coverage and causal source checks."""
import copy
import csv
import datetime as dt
import hashlib
import json
import sqlite3
from collections import defaultdict
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch

import requests

from experiments.e274_birth_source import CACHE, prior_records, project
from f1.data import BASE, ROOT, _s


def main():
    directory = ROOT / 'results'
    summary = json.loads((directory / 'e274_birth_source_summary.json').read_text())
    saved = json.loads((directory / 'e274_birth_projection.json').read_text())
    saved_prior = json.loads((directory / 'e274_prior_birth_records.json').read_text())
    original_hash = hashlib.sha256(CACHE.read_bytes()).hexdigest()
    assert original_hash == summary['original_cache_sha256']
    with (ROOT / 'data/results.csv').open() as file:
        official = [dict(season=int(r['season']), round=int(r['round']), driver=r['driver'], date=r['date'])
                    for r in csv.DictReader(file) if 2018 <= int(r['season']) <= 2022]
    keys = {(r['season'], r['round'], r['driver']): r['date'] for r in official}
    assert len(keys) == len(official) == 2060 and len(summary['pages']) == 24
    output = []; source_seen = set(); available = 0; raw_pages = []
    with sqlite3.connect(f'file:{CACHE}?mode=ro', uri=True) as database, \
            patch('requests.adapters.HTTPAdapter.send', side_effect=AssertionError('audit HTTP forbidden')) as sends:
        expected_pages = [(y, offset, n) for y, n in [(2018, 420), (2019, 420), (2020, 340), (2021, 440), (2022, 440)]
                          for offset in range(0, n, 100)]
        for receipt, (year, offset, total) in zip(summary['pages'], expected_pages):
            assert (receipt['season'], receipt['offset']) == (year, offset)
            req = requests.Request('GET', f'{BASE}/{year}/results.json', params={'offset': offset, 'limit': 100}).prepare()
            key = _s.cache.create_key(req)
            assert key == receipt['cache_key']
            found = database.execute('SELECT value FROM responses WHERE key=?', (key,)).fetchone()
            if found is None:
                assert not receipt['available'] and receipt['reason'] == 'not_cached'
                continue
            response = _s.cache.responses.deserialize(key, found[0])
            if not receipt['available']:
                assert 'metadata_error_type' in receipt
                continue
            available += 1
            body = response.content
            assert hashlib.sha256(body).hexdigest() == receipt['body_sha256']
            assert hashlib.sha256(found[0]).hexdigest() == receipt['blob_sha256'] and len(body) == receipt['body_bytes']
            url = urlsplit(response.url)
            assert response.status_code == receipt['status'] == 200 and response.from_cache
            assert url.scheme == 'https' and url.hostname == 'api.jolpi.ca'
            assert url.path in [f'/ergast/f1/{year}/results.json', f'/ergast/f1/{year}/results/']
            assert parse_qs(url.query) == {'limit': ['100'], 'offset': [str(offset)]}
            assert response.url == receipt['url']
            data = json.loads(body)['MRData']
            assert [int(data[k]) for k in ['limit', 'offset', 'total']] == [100, offset, total]
            table = data['RaceTable']; assert int(table['season']) == year
            raw_pages.append(table['Races'])
            for race in table['Races']:
                assert int(race['season']) == year
                event_date = dt.datetime.strptime(race['date'], '%Y-%m-%d').date()
                for entry in race['Results']:
                    driver = entry['Driver']['driverId']; key = year, int(race['round']), driver
                    assert key in keys and keys[key] == race['date'] and key not in source_seen
                    source_seen.add(key)
                    dob = entry['Driver'].get('dateOfBirth')
                    if dob is None or dob == '':
                        dob = None
                    else:
                        assert isinstance(dob, str) and len(dob) == 10
                        birth = dt.datetime.strptime(dob, '%Y-%m-%d').date()
                        assert birth.isoformat() == dob and birth < event_date
                        assert 15*365.2425 <= (event_date-birth).days <= 60*365.2425
                    output.append(dict(season=year, round=key[1], date=race['date'], driver=driver, dateOfBirth=dob))
        assert sends.call_count == 0
    assert output == saved
    observed = {(r['season'], r['round'], r['driver']): r for r in output}
    birth_sets = defaultdict(set)
    for row in official:
        birth_sets[row['driver']]
    for row in output:
        if row['dateOfBirth']:
            birth_sets[row['driver']].add(row['dateOfBirth'])
    assert summary['official_entries'] == 2060 and summary['source_rows'] == len(output)
    assert summary['source_events'] == len({(r['season'], r['round']) for r in output})
    assert summary['unique_drivers'] == len(birth_sets)
    assert summary['unique_covered'] == sum(len(v) == 1 for v in birth_sets.values())
    conflicts = {d: sorted(v) for d, v in birth_sets.items() if len(v) > 1}
    assert summary['conflicts'] == conflicts
    assert summary['missing_entries'] == [r for r in official if (r['season'], r['round'], r['driver']) not in observed]
    assert summary['missing_birth_dates'] == [r for r in output if r['dateOfBirth'] is None]
    for year in range(2018, 2023):
        cohort = [r for r in official if r['season'] == year]
        covered = sum(observed.get((r['season'], r['round'], r['driver']), {}).get('dateOfBirth') is not None for r in cohort)
        assert summary['years'][str(year)] == dict(entries=len(cohort), covered=covered, coverage=covered/len(cohort))
    independently = []
    for target in [r for r in official if r['season'] >= 2020]:
        event = target['season'], target['round']
        date = dt.datetime.strptime(target['date'], '%Y-%m-%d').date()
        cutoff = date-dt.timedelta(days=(date.isoweekday()-4) % 7)
        earlier = {r['dateOfBirth'] for r in output if r['driver'] == target['driver'] and r['dateOfBirth']
                   and (r['season'], r['round']) < event and dt.datetime.strptime(r['date'], '%Y-%m-%d').date() < cutoff}
        independently.append(dict(**target, prior_dob=sorted(earlier)[0] if len(earlier) == 1 else None, prior_conflict=len(earlier) > 1))
    assert independently == saved_prior and len(saved_prior) == 1220
    for year in [2020, 2021, 2022]:
        rows = [r for r in independently if r['season'] == year]; covered = sum(r['prior_dob'] is not None for r in rows)
        assert summary['prior_years'][str(year)] == dict(entries=len(rows), covered=covered, coverage=covered/len(rows))
    content = not conflicts and summary['unique_covered']/len(birth_sets) >= .95
    content = content and all(v['coverage'] >= .95 for v in summary['years'].values()) and all(v['coverage'] >= .95 for v in summary['prior_years'].values())
    assert summary['content_gate'] == content and summary['source_gate'] == bool(available == 24 and content)
    # Poison every ignored field in all ACTUAL cached records, then reproject.
    poisoned_rows = []; future_rows = []
    target_event = (2022, 10)
    for races in raw_pages:
        poison = copy.deepcopy(races); future = copy.deepcopy(races)
        for race in poison:
            for field in list(race):
                if field not in ['season', 'round', 'date', 'Results']:
                    race[field] = {'poison': [999]}
            for result in race['Results']:
                for field in list(result):
                    if field != 'Driver':
                        result[field] = {'poison': [999]}
                for field in list(result['Driver']):
                    if field not in ['driverId', 'dateOfBirth']:
                        result['Driver'][field] = {'poison': [999]}
        poisoned_rows.extend(project(poison, keys))
        for race in future:
            if (int(race['season']), int(race['round'])) >= target_event:
                for result in race['Results']:
                    result['Driver']['dateOfBirth'] = '1999-01-01'
        future_rows.extend(project(future, keys))
    assert poisoned_rows == output
    targets = [r for r in official if (r['season'], r['round']) <= target_event]
    expected = prior_records(output, targets)
    assert prior_records(future_rows, targets) == expected
    assert prior_records([r for r in output if (r['season'], r['round']) < target_event], targets) == expected
    target_drivers = {r['driver'] for r in targets if (r['season'], r['round']) == target_event}
    selected = sorted(target_drivers)[0]; changed = copy.deepcopy(output)
    for row in changed:
        if row['driver'] == selected and (row['season'], row['round']) < target_event and row['dateOfBirth']:
            date = dt.datetime.strptime(row['dateOfBirth'], '%Y-%m-%d').date()+dt.timedelta(days=1)
            row['dateOfBirth'] = date.isoformat()
    assert prior_records(changed, targets) != expected
    for name, digest in summary['artifact_hashes'].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256(CACHE.read_bytes()).hexdigest() == original_hash and summary['original_cache_unchanged']
    assert summary['actual_http_sends'] == 0
    assert not any(summary[k] for k in ['historical_publication_time_verified', 'winner_fit', 'outer_evaluated', 'evaluated_2026'])
    audit = dict(pages_verified=available, official_entries=2060, inner_prior_entries=1220,
                 source_gate=summary['source_gate'], raw_source_poison_prefix_earlier_controls=True,
                 original_cache_unchanged=True, actual_http_sends=0,
                 output_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('e274_*.json')
                                if p.name != 'e274_independent_audit.json'})
    (directory / 'e274_independent_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print('E274 independent ALL24 readonly raw pages/2060 entries/1220 strict-prior records/date/count/gate/hashes/actual poison/prefix/earlier controls PASS')


if __name__ == '__main__':
    main()
