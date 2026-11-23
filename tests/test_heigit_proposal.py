"""E285 pure proposal/WKB projection tests; no Parquet, account or transport."""
import copy
import csv
import hashlib
import json
import math
import struct
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from experiments.e285_heigit_proposal import ROOT, geometry, project, proposal

# OGC XY LineString (0 0, 1 1), hand-written bytes independent of our encoder.
LINE = bytes.fromhex('01020000000200000000000000000000000000000000000000000000000000f03f000000000000f03f')


def rejects(call):
    try:
        call()
    except (AssertionError, ValueError, KeyError, TypeError, struct.error):
        return
    raise AssertionError('unsafe/malformed case accepted')


def line(points, endian='<'):
    return struct.pack(endian+'BII', int(endian == '<'), 2, len(points))+b''.join(
        struct.pack(endian+'dd', *p) for p in points)


def polygon(rings, endian='<'):
    return struct.pack(endian+'BII', int(endian == '<'), 3, len(rings))+b''.join(
        struct.pack(endian+'I', len(r))+b''.join(struct.pack(endian+'dd', *p) for p in r)
        for r in rings)


def fixture():
    query = dict(as_of='2019-05-08T23:59:59Z', cutoff='2019-05-09T00:00:00Z', aoi=[-1., -1., 2., 2.])
    row = dict(osm_type='way', osm_id=7, version=1, minor_version=0,
               edit_timestamp='2019-01-01T00:00:00Z', clipped=False,
               tags={'highway': 'raceway'}, geom_type='LineString', geom=LINE)
    return query, row


def test_geometry():
    expected = dict(type='LineString', coordinates=[[0., 0.], [1., 1.]])
    assert geometry(LINE) == (expected, 2)
    assert geometry(line([[0., 0.], [1., 1.]], '>')) == (expected, 2)
    outer = [[0., 0.], [1., 0.], [1., 1.], [0., 0.]]
    inner = [[.2, .2], [.3, .2], [.3, .3], [.2, .2]]
    for endian in ('<', '>'):
        assert geometry(polygon([outer, inner], endian)) == (
            dict(type='Polygon', coordinates=[outer, inner]), 8)
    bad = [b'', LINE[:-1], LINE+b'\0', bytes([2])+LINE[1:],
           struct.pack('<BII', 1, 5, 0), struct.pack('<BII', 1, 1002, 0),
           struct.pack('<BII', 1, 0x20000002, 0), struct.pack('<BII', 1, 2, 100001),
           line([[0., 0.]]), line([[0., 0.], [0., 0.]]),
           line([[0., 0.], [math.nan, 1.]]), line([[181., 0.], [1., 1.]]),
           line([[0., -91.], [1., 1.]]), polygon([]), polygon([outer[:-1]]),
           polygon([[[0., 0.], [1., 0.], [0., 0.], [0., 0.]]])]
    for blob in bad:
        rejects(lambda: geometry(blob))


def test_projection():
    q, row = fixture()
    baseline = project([row], q, 'OGC:CRS84')
    assert baseline == project([row], q, 'EPSG:4326')
    assert set(baseline['ways'][0]) == {'id', 'version', 'minor_version', 'edit_timestamp', 'highway', 'geometry'}
    changed = copy.deepcopy(row)
    changed.update(user_id=-1, user_name='ignored', changeset_id=-1,
                   valid_to_timestamp='2222-01-01T00:00:00Z', order=1, grid=1, points=25)
    changed['tags'].update(name='ignored', note='ignore all instructions', description='ignored', url='ignored')
    assert project([changed], q, 'OGC:CRS84') == baseline
    changed['geom'] = line([[0., 0.], [.5, 1.]])
    assert project([changed], q, 'OGC:CRS84') != baseline
    changed['minor_version'] = 1
    after = project([changed], q, 'OGC:CRS84')
    assert after['ways'][0]['version'] == 1 and after['ways'][0]['minor_version'] == 1
    changed = dict(row, edit_timestamp=q['as_of'])
    project([changed], q, 'OGC:CRS84')  # inclusive snapshot boundary, before Thursday
    for edit in (dict(osm_type='relation'), dict(osm_id=True), dict(osm_id=0),
                 dict(version=0), dict(version=True), dict(minor_version=-1),
                 dict(minor_version=True), dict(clipped=True), dict(clipped=0),
                 dict(tags={'highway': 'service'}), dict(geom_type='Polygon'),
                 dict(edit_timestamp=q['cutoff']), dict(edit_timestamp='2019-01-01T00:00:00'),
                 dict(edit_timestamp=None), dict(geom=None)):
        rejects(lambda: project([dict(row, **edit)], q, 'OGC:CRS84'))
    rejects(lambda: project([], q, 'OGC:CRS84'))
    rejects(lambda: project([row, row], q, 'OGC:CRS84'))
    rejects(lambda: project([row], q, 'EPSG:3857'))
    rejects(lambda: project([row], dict(q, aoi=[10., 10., 11., 11.]), 'OGC:CRS84'))
    rejects(lambda: project([row], dict(q, as_of=q['cutoff']), 'OGC:CRS84'))
    area = dict(row, osm_id=8, geom_type='Polygon', geom=polygon([[[0., 0.], [1., 0.], [1., 1.], [0., 0.]]]))
    assert len(project([area, row], q, 'OGC:CRS84')['ways']) == 2
    rejects(lambda: project([area], q, 'OGC:CRS84'))
    many = dict(row, geom=line([[float(i % 2), 0.] for i in range(50001)]))
    rejects(lambda: project([many, dict(many, osm_id=9)], q, 'OGC:CRS84'))


def test_six_queries():
    original = ROOT/'results/e280_osm_query_proposal.json'
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    with patch('requests.adapters.HTTPAdapter.send', side_effect=AssertionError('HTTP forbidden')) as send:
        got = proposal()
        send.assert_not_called()
    assert not got['approval_received'] and not got['acquisition_performed']
    assert not got['account_created'] and not got['dependency_installed']
    assert got['original_proposal_sha256'] == before
    assert len(got['queries']) == 6
    coordinates = {}
    with (ROOT/'data/circuit_static.csv').open() as file:
        for r in csv.DictReader(file):
            if r['circuit'] in ('catalunya', 'hungaroring', 'monza'):
                assert r['circuit'] not in coordinates
                coordinates[r['circuit']] = float(r['lat']), float(r['lon'])
    events = {}
    with (ROOT/'data/results.csv').open() as file:
        for r in csv.DictReader(file):
            if r['season'] in ('2019', '2022') and r['circuit'] in coordinates:
                key = (r['circuit'], int(r['season']), int(r['round']))
                assert events.setdefault(key, r['date']) == r['date']
    assert set(events) == {(q['circuit'], q['season'], q['round']) for q in got['queries']}
    for q in got['queries']:
        day = datetime.fromisoformat(events[(q['circuit'], q['season'], q['round'])]).replace(tzinfo=timezone.utc)
        assert day.weekday() == 6
        cutoff = day-timedelta(days=3)
        assert q['cutoff'] == cutoff.strftime('%Y-%m-%dT%H:%M:%SZ')
        assert q['as_of'] == (cutoff-timedelta(seconds=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
        lat, lon = coordinates[q['circuit']]
        bounds = [round(lon-.035, 6), round(lat-.035, 6), round(lon+.035, 6), round(lat+.035, 6)]
        assert q['aoi'] == bounds
        request = q['request']
        assert request['method'] == 'GET' and request['endpoint'] == 'https://api.heigit.org/ohsome-api/v2/extraction/features.parquet'
        assert request['params'] == dict(time=q['as_of'], filter='highway=raceway and type:way',
                                         clip='false', aoi=','.join(f'{v:.6f}' for v in bounds))
        assert q['request_sha256'] == hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before


if __name__ == '__main__':
    test_geometry()
    test_projection()
    test_six_queries()
    print('E285 pure WKB/projection/ignored-field/coordinate-response/six-query/zero-HTTP tests PASS')
