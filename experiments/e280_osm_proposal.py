"""E280 offline-only query proposal and synthetic-tested historical projection.

No HTTP client or acquisition entry point: explicit ODbL approval is pending.
"""
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone

from f1.data import DATA, ROOT

SITES = ['catalunya', 'hungaroring', 'monza']
ENDPOINT = 'https://overpass-api.de/api/interpreter'


def utc(value):
    assert isinstance(value, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', value)
    return datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)


def manifest():
    coordinates = {}
    with (DATA / 'circuit_static.csv').open() as file:
        for row in csv.DictReader(file):
            if row['circuit'] in SITES:
                assert row['circuit'] not in coordinates
                coordinates[row['circuit']] = float(row['lat']), float(row['lon'])
    assert set(coordinates) == set(SITES)
    events = set()
    with (DATA / 'results.csv').open() as file:
        for row in csv.DictReader(file):
            if row['season'] not in ['2019', '2022'] or row['circuit'] not in SITES:
                continue
            events.add((row['circuit'], int(row['season']), int(row['round']), row['date']))
    assert len(events) == 6 and {(s, y) for s, y, r, d in events} == {(s, y) for s in SITES for y in [2019, 2022]}
    queries = []
    for circuit, year, rnd, date in sorted(events):
        race_date = datetime.strptime(date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        cutoff = race_date-timedelta(days=(race_date.weekday()-3) % 7)
        as_of = cutoff-timedelta(seconds=1)
        stamp = as_of.strftime('%Y-%m-%dT%H:%M:%SZ'); cutoff_stamp = cutoff.strftime('%Y-%m-%dT%H:%M:%SZ')
        lat, lon = coordinates[circuit]
        bbox = [round(lat-.035, 6), round(lon-.035, 6), round(lat+.035, 6), round(lon+.035, 6)]
        box = ','.join(f'{v:.6f}' for v in bbox)
        query = f'[out:json][timeout:25][maxsize:67108864][date:"{stamp}"];way["highway"="raceway"]({box});(._;>;);out meta;'
        queries.append(dict(circuit=circuit, season=year, round=rnd, race_date=date, cutoff=cutoff_stamp, as_of=stamp,
                            bbox=bbox, endpoint=ENDPOINT, query=query, query_sha256=hashlib.sha256(query.encode()).hexdigest()))
    return dict(approval_received=False, acquisition_performed=False, licence='ODbL-1.0', queries=queries,
                source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [DATA / 'circuit_static.csv', DATA / 'results.csv']})


def project(body, as_of, cutoff):
    assert utc(as_of) < utc(cutoff)
    assert isinstance(body, dict) and not body.get('remark')
    elements = body.get('elements'); assert isinstance(elements, list) and 0 < len(elements) <= 100000
    nodes = {}; ways = {}; seen = set()
    for element in elements:
        assert isinstance(element, dict) and element.get('type') in ['node', 'way']
        kind = element['type']; ident = element.get('id'); version = element.get('version')
        assert type(ident) is int and ident > 0 and type(version) is int and version > 0
        assert (kind, ident) not in seen; seen.add((kind, ident))
        stamp = element.get('timestamp'); assert utc(stamp) <= utc(as_of) and utc(stamp) < utc(cutoff)
        record = dict(id=ident, version=version, timestamp=stamp)
        if kind == 'node':
            lat, lon = element.get('lat'), element.get('lon')
            assert type(lat) in [int, float] and type(lon) in [int, float]
            assert math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180
            nodes[ident] = dict(record, lat=lat, lon=lon)
        else:
            assert isinstance(element.get('tags'), dict) and element['tags'].get('highway') == 'raceway'
            refs = element.get('nodes')
            assert isinstance(refs, list) and len(refs) >= 2 and all(type(n) is int and n > 0 for n in refs)
            assert len(set(refs)) >= 2
            ways[ident] = dict(record, nodes=refs.copy(), highway='raceway')
    assert ways and len(nodes) >= 2
    referenced = {n for way in ways.values() for n in way['nodes']}
    assert referenced == set(nodes)
    return dict(nodes=[nodes[n] for n in sorted(nodes)], ways=[ways[n] for n in sorted(ways)])


def main():
    path = ROOT / 'results/e280_osm_query_proposal.json'
    assert not path.exists(), 'do not overwrite registered proposal'
    proposal = manifest(); path.write_text(json.dumps(proposal, indent=2)+'\n')
    print('E280 six offline queries prepared; approval pending; HTTP requests ZERO')


if __name__ == '__main__':
    main()
