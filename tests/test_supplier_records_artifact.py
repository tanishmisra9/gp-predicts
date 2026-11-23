"""Independent E270 receipts, offline identities, full-entry joins and counts."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

import fastf1
import yaml
import f1.ff1
from fastf1.internals.f1account import LivetimingAuthAdapter

from experiments.e270_supplier_records import project
from f1.data import ROOT
from f1.profile import PU


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = ROOT / 'results'
    manifest = json.loads((result / 'e269_supplier_source_manifest.json').read_text())
    metadata = manifest['race_file_metadata'] + manifest['entrant_file_metadata']
    saved = json.loads((result / 'e270_projection.json').read_text())
    receipts = json.loads((result / 'e270_receipts.json').read_text())
    reports = json.loads((result / 'e270_identity_audit.json').read_text())
    summary = json.loads((result / 'e270_summary.json').read_text())
    prior = {(r['season'], r['round']): r['timing_cache_sha256'] for r in
             json.loads((result / 'e266_source_status.json').read_text())}
    assert len(metadata) == len(saved) == len(receipts) == 108
    assert len(reports) == 103
    cache = Path('/tmp/gp-predicts-e270-source')
    before = {p.name: sha(p) for p in cache.glob('*.yml')}
    projected = []
    for meta, output, receipt in zip(metadata, saved, receipts):
        raw = (cache / (meta['git_blob'] + '.yml')).read_bytes()
        blob = hashlib.sha1(b'blob %d\0' % len(raw) + raw).hexdigest()
        assert blob == meta['git_blob']
        assert receipt == dict(meta, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        rows = yaml.safe_load(raw)
        fields = (['driverId', 'driverNumber', 'constructorId', 'engineManufacturerId', 'engineId']
                  if 'round' in meta else ['entrantId', 'constructorId', 'engineManufacturerId', 'engineId'])
        expected = []
        for row in rows:
            retained = {}
            for field in fields:
                value = row.get(field)
                retained[field] = (str(int(value)) if field == 'driverNumber' and value is not None
                                   else (value.strip() or None) if isinstance(value, str) else None)
            expected.append(retained)
        assert output == dict(meta, rows=expected)
        poisoned = [{k: v if k in fields else {'outcome_grid_text_poison': [None, 999]}
                     for k, v in row.items()} for row in reversed(rows)]
        assert project(yaml.safe_dump(poisoned), 'round' not in meta) == list(reversed(expected))
        if rows:
            changed = [dict(row) for row in rows]
            changed[0]['engineManufacturerId'] = 'e270-positive-source-control'
            assert project(yaml.safe_dump(changed), 'round' not in meta) != expected
        projected.append(expected)
    # Independently parse the official entry denominator without importing its loader.
    with (ROOT / 'data/results.csv').open() as file:
        official = [r for r in csv.DictReader(file) if 2018 <= int(r['season']) <= 2022]
    entries = {}
    for row in official:
        key = int(row['season']), int(row['round']), row['code']
        assert key not in entries
        entries[key] = row['team']
    with (result / 'e270_matched.csv').open() as file:
        matched_rows = list(csv.DictReader(file))
        matched = {(int(r['season']), int(r['round']), r['code']): r for r in matched_rows}
    assert len(matched_rows) == len(matched)
    assert set(entries) == set(matched) and len(entries) == len(official) == 2060
    annual = {m['season']: {(r['constructorId'], r['engineManufacturerId']) for r in rows
                            if r['constructorId'] and r['engineManufacturerId']}
              for m, rows in zip(metadata[103:], projected[103:])}
    abbreviations = {'mer': 'mercedes', 'fer': 'ferrari', 'ren': 'renault',
                     'hon': 'honda', 'rbf': 'red-bull-ford', 'aud': 'audi'}
    counts = defaultdict(Counter)
    teams = defaultdict(Counter)
    brands = defaultdict(set)
    missing = []; disagreements = []; extras = 0; contradictions = 0
    timing_paths = {}
    fastf1.Cache.offline_mode(True)
    with patch('requests.adapters.HTTPAdapter.send', side_effect=AssertionError('audit HTTP forbidden')) as http, \
            patch.object(LivetimingAuthAdapter, '_get_token', side_effect=AssertionError('token lookup forbidden')) as auth:
        for meta, rows, report in zip(metadata[:103], projected[:103], reports):
            year, rnd = meta['season'], meta['round']
            assert (report['season'], report['round']) == (year, rnd)
            session = fastf1.get_session(year, rnd, 'Q')
            timing = Path(fastf1.Cache._get_cache_file_path(session.api_path, '_extended_timing_data'))
            digest = sha(timing)
            assert digest == prior[year, rnd] == report['timing_sha256']
            timing_paths[timing] = digest
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            number_codes = defaultdict(set); code_numbers = defaultdict(set)
            for number, code in zip(session.laps.DriverNumber, session.laps.Driver):
                if str(number) in ('nan', 'None', '') or str(code) in ('nan', 'None', ''):
                    continue
                number = str(int(number))
                number_codes[number].add(code); code_numbers[code].add(number)
            assert all(len(v) == 1 for v in number_codes.values())
            assert all(len(v) == 1 for v in code_numbers.values())
            identity = {n: next(iter(v)) for n, v in number_codes.items()}
            assert identity == report['identities']
            by_code = {}; unmatched = []; associations = defaultdict(set)
            for row in rows:
                number = row['driverNumber']
                code = identity.get(number)
                if row['constructorId'] and row['engineManufacturerId']:
                    associations[row['constructorId']].add(row['engineManufacturerId'])
                if (year, rnd, code) not in entries:
                    unmatched.append(row)
                else:
                    assert code not in by_code
                    by_code[code] = row
            conflicts = {k for k, v in associations.items() if len(v) > 1}
            assert conflicts == set(report['conflicts']) and unmatched == report['unmatched']
            extras += len(unmatched); contradictions += bool(conflicts)
            for (y, r, code), team in entries.items():
                if (y, r) != (year, rnd):
                    continue
                source = by_code.get(code, {})
                brand = source.get('engineManufacturerId')
                old = None
                for (start, end), mapping in PU.items():
                    if start <= year <= end:
                        old = mapping.get(team); break
                covered = bool(brand and source.get('constructorId') and source.get('driverId'))
                comparable = bool(brand and old and brand in abbreviations.values())
                flags = dict(covered=covered, comparable=comparable,
                             disagreement=bool(comparable and abbreviations[old] != brand),
                             annual_consistent=bool(brand and (source.get('constructorId'), brand) in annual[year]))
                output = matched[year, rnd, code]
                assert output['team'] == team and output['legacy'] == (old or '')
                for field in ['driverId', 'driverNumber', 'constructorId', 'engineManufacturerId', 'engineId']:
                    assert output[field] == (source.get(field) or '')
                assert all(output[k] == str(v) for k, v in flags.items())
                row_counts = dict(entries=1, **{k: int(v) for k, v in flags.items()},
                                  legacy_unknown=int(old is None),
                                  noncomparable_covered_legacy_known=int(covered and old is not None and not comparable))
                counts[year].update(row_counts); teams[year, team].update(row_counts)
                if brand:
                    brands[year, team].add(brand)
                if not covered:
                    missing.append(dict(season=year, round=rnd, code=code, team=team))
                if flags['disagreement']:
                    disagreements.append(dict(season=year, round=rnd, code=code, team=team,
                                              legacy=old, recorded_brand=brand))
    assert http.call_count == auth.call_count == 0
    for year, stats in counts.items():
        expected = dict(stats, coverage=stats['covered'] / stats['entries'])
        assert expected == summary['years'][str(year)]
    assert summary['team_brands'] == [dict(season=y, team=t, engineManufacturerId=sorted(brands[y, t]))
                                     for y, t in sorted(teams)]
    gate = len(receipts) == 108 and all(c['covered'] / c['entries'] >= .95 for c in counts.values()) and not contradictions
    assert summary['unmatched_source_rows'] == extras and summary['conflicting_events'] == contradictions
    assert summary['recorded_brand_source_gate'] == gate
    assert not any(summary[k] for k in ['physical_oem_mapping_verified', 'historical_publication_time_verified',
                                       'winner_fit', 'production_changed'])
    assert before == {p.name: sha(p) for p in cache.glob('*.yml')}
    assert all(sha(path) == digest for path, digest in timing_paths.items())
    audit = dict(files_verified=108, sessions_verified=103, official_entries=2060,
                 http_sends=http.call_count, token_lookups=auth.call_count,
                 source_and_timing_cache_immutable=True, recorded_brand_source_gate=gate,
                 per_team_counts=[dict(season=y, team=t, **teams[y, t]) for y, t in sorted(teams)],
                 missing_official_entries=missing, comparable_disagreements=disagreements,
                 artifact_sha256={p.name: sha(p) for p in sorted(result.glob('e270_*'))
                                  if p.suffix in ('.json', '.csv') and p.name != 'e270_independent_audit.json'})
    (result / 'e270_independent_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('E270 independent ALL108 hashes/projections/poisoning, ALL103 offline identities, ALL2060 joins/counts/cache immutability PASS')


if __name__ == '__main__':
    main()
