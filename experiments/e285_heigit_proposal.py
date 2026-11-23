"""E285 offline proposal/projector only: no account, dependency or HTTP action."""
import hashlib
import json
import math
import struct
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = 'https://api.heigit.org/ohsome-api/v2/extraction/features.parquet'
MAX_COORDS = 100000


def utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    assert isinstance(value, datetime) and value.tzinfo is not None
    assert value.utcoffset().total_seconds() == 0
    return value.astimezone(timezone.utc)


def proposal():
    original = ROOT/'results/e280_osm_query_proposal.json'
    old = json.loads(original.read_text())
    queries = []
    for q in old['queries']:
        south, west, north, east = q['bbox']
        aoi = [west, south, east, north]
        assert utc(q['as_of']) < utc(q['cutoff'])
        params = dict(aoi=','.join(f'{v:.6f}' for v in aoi), time=q['as_of'],
                      filter='highway=raceway and type:way', clip='false')
        request = dict(method='GET', endpoint=ENDPOINT, params=params)
        digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        queries.append(dict(circuit=q['circuit'], season=q['season'], round=q['round'],
                            as_of=q['as_of'], cutoff=q['cutoff'], aoi=aoi,
                            request=request, request_sha256=digest))
    assert len(queries) == 6 and len({q['request_sha256'] for q in queries}) == 6
    return dict(experiment='E285', registration_commit='6ed18e3', approval_received=False,
                acquisition_performed=False, account_created=False, dependency_installed=False,
                queries=queries, original_proposal_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                required_new_authorization=['User-owned free HeiGIT account/key setup or existing-account reuse',
                                            'Secure key use only for these requests; no .env/other secret lookup',
                                            'Pinned PyArrow26.0.1 Apache-2.0 wheel in research venv',
                                            'Six additional scoped HeiGIT snapshot requests'],
                limits=dict(max_attempts=6, start_spacing_seconds=5, connect_timeout_seconds=10,
                            read_timeout_seconds=35, max_body_bytes=2*1024**2,
                            max_total_body_bytes=12*1024**2, max_declared_selected_uncompressed_bytes=16*1024**2,
                            max_rows=10000, max_coordinates=MAX_COORDS,
                            retries=0, redirects=0, global_stop_on_any_failure=True),
                attribution='© OpenStreetMap contributors; ohsome platform by HeiGIT',
                licence='ODbL-1.0; existing E280 approval does not authorize these additional actions')


def geometry(blob):
    """Complete 2D OGC WKB line/polygon -> explicit geographic coordinates."""
    assert isinstance(blob, bytes) and len(blob) >= 9 and blob[0] in (0, 1)
    endian = '<' if blob[0] else '>'
    cursor = 1
    def take(fmt):
        nonlocal cursor
        size = struct.calcsize(endian+fmt)
        assert cursor+size <= len(blob)
        value = struct.unpack_from(endian+fmt, blob, cursor)
        cursor += size
        return value
    kind, = take('I')
    assert kind in (2, 3), 'only complete 2D LineString/Polygon, no EWKB/SRID/Z/M or collections'
    count = 1 if kind == 2 else take('I')[0]
    assert 1 <= count <= MAX_COORDS//4
    parts = []; total = 0
    for _ in range(count):
        n, = take('I')
        assert (2 if kind == 2 else 4) <= n <= MAX_COORDS-total
        assert cursor+16*n <= len(blob)
        points = [list(take('dd')) for _ in range(n)]
        assert all(math.isfinite(x) and math.isfinite(y) and -180 <= x <= 180 and -90 <= y <= 90
                   for x, y in points)
        assert len({tuple(p) for p in points}) >= (2 if kind == 2 else 3)
        if kind == 3:
            assert points[0] == points[-1]
        parts.append(points); total += n
    assert cursor == len(blob)
    return dict(type='LineString' if kind == 2 else 'Polygon',
                coordinates=parts[0] if kind == 2 else parts), total


def project(records, query, crs):
    """Decoded allowlisted records + independently verified file CRS -> projection.

    Reject an entire malformed snapshot; never skip an unknown record. The
    caller must verify native Parquet CRS evidence before calling this projector.
    """
    assert crs in ('OGC:CRS84', 'EPSG:4326')
    snapshot, cutoff = utc(query['as_of']), utc(query['cutoff'])
    assert snapshot < cutoff
    assert isinstance(records, list) and 0 < len(records) <= 10000
    west, south, east, north = query['aoi']
    assert -180 <= west < east <= 180 and -90 <= south < north <= 90
    result = []; seen = set(); total = 0
    for row in records:
        assert isinstance(row, dict) and row['osm_type'] == 'way'
        ident, version, minor = row['osm_id'], row['version'], row['minor_version']
        assert type(ident) is int and ident > 0 and ident not in seen
        assert type(version) is int and version > 0 and type(minor) is int and minor >= 0
        seen.add(ident)
        assert row['clipped'] is False and row['tags']['highway'] == 'raceway'
        stamp = utc(row['edit_timestamp'])
        assert stamp <= snapshot and stamp < cutoff
        shape, count = geometry(row['geom'])
        assert shape['type'] == row['geom_type']
        total += count
        assert total <= MAX_COORDS
        points = shape['coordinates'] if shape['type'] == 'LineString' else [p for ring in shape['coordinates'] for p in ring]
        xs, ys = zip(*points)
        assert min(xs) <= east and max(xs) >= west and min(ys) <= north and max(ys) >= south
        result.append(dict(id=ident, version=version, minor_version=minor,
                           edit_timestamp=stamp.isoformat().replace('+00:00', 'Z'),
                           highway='raceway', geometry=shape))
    assert any(r['geometry']['type'] == 'LineString' for r in result)
    return dict(crs='OGC:CRS84', as_of=query['as_of'], ways=sorted(result, key=lambda r: r['id']))


if __name__ == '__main__':
    path = ROOT/'results/e285_heigit_query_proposal.json'
    assert not path.exists(), 'never overwrite a registered proposal'
    path.write_text(json.dumps(proposal(), indent=2)+'\n')
    print('Six offline E285 requests prepared; no account/key/dependency/map action; approval pending')
