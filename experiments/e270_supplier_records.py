"""Approved historical F1DB recorded-brand audit; no features or winner fits."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import pandas as pd
import yaml

from experiments.e269_supplier_source_audit import PIN, REPO, metadata_scope
from f1.data import ROOT, load_results
from f1.profile import PU

CACHE = Path('/tmp/gp-predicts-e270-source')
RACE_FIELDS = ['driverId', 'driverNumber', 'constructorId', 'engineManufacturerId', 'engineId']
ENTRANT_FIELDS = ['entrantId', 'constructorId', 'engineManufacturerId', 'engineId']
LEGACY = dict(mer='mercedes', fer='ferrari', ren='renault', hon='honda',
              rbf='red-bull-ford', aud='audi')


def verified(raw, digest):
    assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == digest
    return raw


def acquire(meta):
    CACHE.mkdir(exist_ok=True)
    dest = CACHE / (meta['git_blob'] + '.yml')
    if dest.exists():
        return verified(dest.read_bytes(), meta['git_blob'])
    response = json.loads(subprocess.check_output(
        ['gh', 'api', f"repos/{REPO}/git/blobs/{meta['git_blob']}"]))
    assert response['sha'] == meta['git_blob'] and response['encoding'] == 'base64'
    raw = verified(base64.b64decode(response['content']), meta['git_blob'])
    dest.write_bytes(raw)
    return raw


def project(raw, entrant=False):
    data = yaml.safe_load(raw)
    assert isinstance(data, list) and all(isinstance(r, dict) for r in data)
    fields = ENTRANT_FIELDS if entrant else RACE_FIELDS
    out = []
    for row in data:
        r = {key: row.get(key) for key in fields}
        for key, value in r.items():
            if key == 'driverNumber' and value is not None:
                assert isinstance(value, (int, str)) and not isinstance(value, bool)
                assert str(value).isdigit()
                r[key] = str(int(value))
            else:
                assert value is None or isinstance(value, str), key
                r[key] = value.strip() or None if isinstance(value, str) else None
        out.append(r)
    if not entrant:
        numbers = [r['driverNumber'] for r in out if r['driverNumber'] is not None]
        assert len(numbers) == len(set(numbers)), 'duplicate source car number'
    return out


def identities(laps):
    pairs = laps[['DriverNumber', 'Driver']].dropna().drop_duplicates().copy()
    pairs['DriverNumber'] = pairs.DriverNumber.map(lambda v: str(int(v)))
    assert not pairs.DriverNumber.duplicated().any(), 'ambiguous number'
    assert not pairs.Driver.duplicated().any(), 'ambiguous code'
    return dict(pairs.itertuples(index=False, name=None))


def main():
    import fastf1
    import f1.ff1  # Existing cache location, no source refresh.
    from fastf1.internals.f1account import LivetimingAuthAdapter
    manifest = json.loads((ROOT / 'results/e269_supplier_source_manifest.json').read_text())
    assert manifest['pin'] == PIN and manifest['repo'] == REPO
    races = manifest['race_file_metadata']; entrants = manifest['entrant_file_metadata']
    records = races + entrants
    check = metadata_scope([dict(type='blob', path=m['path'], sha=m['git_blob']) for m in records])
    assert check == (races, entrants) and len(races) == 103 and len(entrants) == 5
    assert len({m['path'] for m in records}) == 108
    official = load_results().loc[lambda d: d.season.between(2018, 2022),
                                 ['season', 'round', 'code', 'team']].copy()
    assert not official.duplicated(['season', 'round', 'code']).any()
    assert set(map(tuple, official[['season', 'round']].drop_duplicates().to_numpy())) == {
        (m['season'], m['round']) for m in races}
    projections = []; receipts = []; annual = {}
    for m in records:
        raw = acquire(m)
        is_entrant = 'round' not in m
        projected = project(raw, is_entrant)
        projections.append(dict(**m, rows=projected))
        receipts.append(dict(**m, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        if is_entrant:
            annual[m['season']] = {(r['constructorId'], r['engineManufacturerId']) for r in projected
                                    if r['constructorId'] and r['engineManufacturerId']}
    fastf1.Cache.offline_mode(True)
    event_reports = []; joined = []
    for item in projections[:103]:
        year, rnd = item['season'], item['round']
        with patch('requests.adapters.HTTPAdapter.send', side_effect=RuntimeError('E270 ALL HTTP forbidden')), \
                patch.object(LivetimingAuthAdapter, '_get_token', return_value=None):
            session = fastf1.get_session(year, rnd, 'Q')
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            identity = identities(session.laps)
            timing = Path(fastf1.Cache._get_cache_file_path(session.api_path, 'timing_data'))
            timing_hash = hashlib.sha256(timing.read_bytes()).hexdigest()
        event = official[official.season.eq(year) & official['round'].eq(rnd)]
        by_code = {}; unmatched = []
        for row in item['rows']:
            code = identity.get(row['driverNumber'])
            if code is None or code not in set(event.code):
                unmatched.append(row); continue
            assert code not in by_code
            by_code[code] = row
        conflicts = []
        for constructor in {r['constructorId'] for r in item['rows'] if r['constructorId']}:
            brands = {r['engineManufacturerId'] for r in item['rows']
                      if r['constructorId'] == constructor and r['engineManufacturerId']}
            if len(brands) > 1:
                conflicts.append(constructor)
        for y, r, code, team in event.itertuples(index=False, name=None):
            row = by_code.get(code, {})
            brand = row.get('engineManufacturerId')
            old = next((mapping.get(team) for (a, b), mapping in PU.items() if a <= y <= b), None)
            comparable = bool(brand and old and brand in LEGACY.values())
            joined.append(dict(season=y, round=r, code=code, team=team,
                               **{f: row.get(f) for f in RACE_FIELDS},
                               legacy=old, covered=bool(brand and row.get('constructorId') and row.get('driverId')),
                               comparable=comparable,
                               disagreement=bool(comparable and LEGACY[old] != brand),
                               annual_consistent=bool(brand and (row.get('constructorId'), brand) in annual[y])))
        event_reports.append(dict(season=year, round=rnd, timing_sha256=timing_hash,
                                  identities=identity, unmatched=unmatched, conflicts=conflicts))
        print(year, rnd, 'identity audit complete', flush=True)
    table = pd.DataFrame(joined)
    years = {}
    for year, frame in table.groupby('season'):
        years[int(year)] = dict(entries=len(frame), covered=int(frame.covered.sum()),
                               coverage=float(frame.covered.mean()),
                               comparable=int(frame.comparable.sum()),
                               disagreement=int(frame.disagreement.sum()),
                               legacy_unknown=int(frame.legacy.isna().sum()),
                               noncomparable_covered_legacy_known=int((frame.covered & frame.legacy.notna() & ~frame.comparable).sum()),
                               annual_consistent=int(frame.annual_consistent.sum()))
    brands = table.groupby(['season', 'team'], sort=True).engineManufacturerId.agg(
        lambda s: sorted(set(s.dropna()))).reset_index().to_dict('records')
    summary = dict(data_licence_approval_received=True, scope='E269 user-approved 2018–22 supplier audit',
                   pin=PIN, files=len(receipts), years=years, team_brands=brands,
                   unmatched_source_rows=sum(len(e['unmatched']) for e in event_reports),
                   conflicting_events=sum(bool(e['conflicts']) for e in event_reports),
                   recorded_brand_source_gate=all(v['coverage'] >= .95 for v in years.values())
                   and not any(e['conflicts'] for e in event_reports),
                   physical_oem_mapping_verified=False, historical_publication_time_verified=False,
                   winner_fit=False, production_changed=False)
    prefix = ROOT / 'results/e270_'
    for suffix, value in [('projection.json', projections), ('receipts.json', receipts),
                           ('identity_audit.json', event_reports), ('summary.json', summary)]:
        Path(str(prefix) + suffix).write_text(json.dumps(value, indent=2) + '\n')
    table.to_csv(str(prefix) + 'matched.csv', index=False)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
