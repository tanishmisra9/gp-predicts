"""Read-only E285 receipts and independent all-record projection audit; no transport/key."""
import hashlib
import json
import math
import struct
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['osm_type', 'osm_id', 'version', 'minor_version', 'edit_timestamp',
          'geom_type', 'geom', 'clipped', 'tags']


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def stamp(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    assert isinstance(value, datetime) and value.tzinfo and value.utcoffset() == timedelta(0)
    return value.astimezone(timezone.utc)


def shape(wkb):
    """Separate WKB route: integer byte reads + coordinate iter_unpack, no producer calls."""
    assert isinstance(wkb, bytes) and len(wkb) >= 9 and wkb[0] in (0, 1)
    order, cursor = ('little' if wkb[0] else 'big'), 1
    def integer():
        nonlocal cursor
        assert cursor+4 <= len(wkb)
        n = int.from_bytes(wkb[cursor:cursor+4], order)
        cursor += 4
        return n
    kind = integer()
    assert kind in (2, 3)
    rings = 1 if kind == 2 else integer()
    assert 1 <= rings <= 25000
    parts = []; count = 0
    for _ in range(rings):
        n = integer()
        assert (2 if kind == 2 else 4) <= n <= 100000-count
        stop = cursor+16*n
        assert stop <= len(wkb)
        points = [list(p) for p in struct.iter_unpack('<dd' if order == 'little' else '>dd', wkb[cursor:stop])]
        assert len(set(map(tuple, points))) >= (2 if kind == 2 else 3)
        assert all(math.isfinite(x) and math.isfinite(y) and -180 <= x <= 180 and -90 <= y <= 90 for x, y in points)
        if kind == 3:
            assert points[0] == points[-1]
        parts.append(points); count += n; cursor = stop
    assert cursor == len(wkb)
    return dict(type='LineString' if kind == 2 else 'Polygon', coordinates=parts[0] if kind == 2 else parts), count


def snapshot(raw, query):
    """Independent Parquet read/WKB/chronology/counts; shared native parser, no producer imports."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    assert pa.__version__ == '26.0.1'
    assert isinstance(raw, bytes) and 12 <= len(raw) <= 2097152 and raw[:4] == raw[-4:] == b'PAR1'
    cutoff, as_of = stamp(query['cutoff']), stamp(query['as_of'])
    assert cutoff.weekday() == 3 and (cutoff.hour, cutoff.minute, cutoff.second, cutoff.microsecond) == (0, 0, 0, 0)
    assert as_of+timedelta(seconds=1) == cutoff
    with pq.ParquetFile(pa.BufferReader(raw), pre_buffer=False, arrow_extensions_enabled=False) as file:
        meta = file.metadata
        assert 1 <= meta.num_rows <= 10000
        names = file.schema_arrow.names
        assert len(names) == len(set(names)) and set(FIELDS) <= set(names)
        types = [pa.string(), pa.int64(), pa.int32(), pa.int32(), pa.timestamp('us', tz='UTC'),
                 pa.string(), pa.binary(), pa.bool_(), pa.map_(pa.string(), pa.string())]
        assert [file.schema_arrow.field(n).type for n in FIELDS] == types
        leaves = {i: file.schema.column(i) for i in range(len(file.schema))}
        geometry = [c for c in leaves.values() if c.path == 'geom']
        assert len(geometry) == 1 and geometry[0].logical_type.type == 'GEOMETRY'
        definition = json.loads(geometry[0].logical_type.to_json())
        assert definition['Type'] == 'Geometry'
        crs = definition.get('crs', 'OGC:CRS84')
        assert crs in ('OGC:CRS84', 'EPSG:4326')
        selected = [i for i, c in leaves.items() if c.path.split('.')[0] in FIELDS]
        sizes = [meta.row_group(g).column(i).total_uncompressed_size
                 for g in range(meta.num_row_groups) for i in selected]
        assert all(n >= 0 for n in sizes) and sum(sizes) <= 16777216
        table = file.read(columns=FIELDS, use_threads=False, use_pandas_metadata=False)
        assert table.num_rows == meta.num_rows and table.column_names == FIELDS
        assert all(column.null_count == 0 for column in table.columns)
        rows = table.to_pylist()
    ways = {}; coordinates = 0; counts = dict(LineString=0, Polygon=0)
    west, south, east, north = query['aoi']
    assert -180 <= west < east <= 180 and -90 <= south < north <= 90
    for r in rows:
        assert r['osm_type'] == 'way' and r['clipped'] is False
        ident = r['osm_id']
        assert type(ident) is int and ident > 0 and ident not in ways
        assert type(r['version']) is int and r['version'] > 0
        assert type(r['minor_version']) is int and r['minor_version'] >= 0
        tags = r['tags']
        assert isinstance(tags, list) and all(isinstance(k, str) and isinstance(v, str) for k, v in tags)
        assert len(set(k for k, v in tags)) == len(tags)
        assert [v for k, v in tags if k == 'highway'] == ['raceway']
        edit = stamp(r['edit_timestamp'])
        assert edit <= as_of < cutoff
        geometry, n = shape(r['geom'])
        assert geometry['type'] == r['geom_type']
        coordinates += n
        assert coordinates <= 100000
        points = geometry['coordinates'] if geometry['type'] == 'LineString' else [p for ring in geometry['coordinates'] for p in ring]
        assert min(p[0] for p in points) <= east and max(p[0] for p in points) >= west
        assert min(p[1] for p in points) <= north and max(p[1] for p in points) >= south
        counts[geometry['type']] += 1
        ways[ident] = dict(id=ident, version=r['version'], minor_version=r['minor_version'],
                           edit_timestamp=edit.isoformat().replace('+00:00', 'Z'), highway='raceway', geometry=geometry)
    assert counts['LineString'] >= 1
    projection = dict(crs='OGC:CRS84', as_of=query['as_of'], ways=[ways[i] for i in sorted(ways)])
    return dict(projection=projection, rows=len(rows), coordinates=coordinates, types=counts,
                native_crs=crs, declared_selected_uncompressed_bytes=sum(sizes))


def audit(directory):
    """Terminal receipt directory -> six statuses; all source bytes stay unchanged."""
    assert not directory.is_symlink() and not directory.resolve().is_relative_to(ROOT)
    assert not (directory/'acquisition.lock').exists(), 'active/uncertain acquisition must be resolved first'
    plan_raw = (ROOT/'results/e285_heigit_query_proposal.json').read_bytes()
    assert sha(plan_raw) == 'e3fe7f9b688bd87f9095568bc3803a22d22af8b3ff7c6829a860da4c10070851'
    plan = json.loads(plan_raw); queries = plan['queries']
    assert len(queries) == 6
    state_path = directory/'receipts.json'
    assert state_path.exists(), 'acquisition has not started; not a source failure'
    assert not state_path.is_symlink()
    state_raw = state_path.read_bytes(); state = json.loads(state_raw)
    assert state['proposal_sha256'] == sha(json.dumps(plan, sort_keys=True).encode())
    receipts = state['receipts']
    assert isinstance(receipts, list) and len(receipts) <= 6
    result = []; expected = set(); before = {state_path: sha(state_raw)}; raw_total = 0
    for i, q in enumerate(queries):
        assert sha(json.dumps(q['request'], sort_keys=True).encode()) == q['request_sha256']
        row = dict(index=i, circuit=q['circuit'], season=q['season'], round=q['round'], status='unattempted_unknown')
        if i < len(receipts):
            r = receipts[i]
            assert type(r['index']) is int and r['index'] == i and r['request_sha256'] == q['request_sha256']
            assert r['state'] in ('success', 'failed', 'started')
            if r['state'] != 'success':
                assert i == len(receipts)-1, 'global stop violated'
            row['status'] = r['state']
            for field, suffix, hash_field in [('raw_file', '.parquet', 'raw_sha256'),
                                               ('projection_file', '.projection.json', 'projection_sha256')]:
                if field in r:
                    name = q['request_sha256']+suffix
                    assert r[field] == name
                    path = directory/name
                    assert not path.is_symlink() and path.is_file()
                    if field == 'raw_file':
                        assert path.stat().st_size <= 2097152
                    body = path.read_bytes()
                    assert sha(body) == r[hash_field]
                    expected.add(name); before[path] = sha(body)
                    if field == 'raw_file':
                        assert type(r['bytes']) is int and len(body) == r['bytes']; raw_total += len(body)
            if r['state'] == 'success':
                assert type(r['status_code']) is int and r['status_code'] == 200
                assert 'raw_file' in r and 'projection_file' in r
                raw = (directory/r['raw_file']).read_bytes()
                completed = subprocess.run([str(ROOT/'.venv-e285/bin/python'), '-I', str(Path(__file__).resolve()),
                                            '--snapshot', json.dumps(q)], input=raw, stdout=subprocess.PIPE,
                                           stderr=subprocess.DEVNULL, env={}, check=False)
                assert completed.returncode == 0, 'independent snapshot audit failed'
                checked = json.loads(completed.stdout)
                saved = json.loads((directory/r['projection_file']).read_bytes())
                assert saved == checked.pop('projection')
                assert type(r['ways']) is int and r['ways'] == checked['rows'] == sum(checked['types'].values())
                row.update(checked, raw_sha256=r['raw_sha256'], projection_sha256=r['projection_sha256'])
        result.append(row)
    present = {p.name for pattern in ('*.parquet', '*.projection.json') for p in directory.glob(pattern)}
    assert present == expected and raw_total <= 12582912
    notice = directory/'SOURCE_NOTICE.txt'
    assert notice.is_file() and not notice.is_symlink()
    text = notice.read_text()
    assert all(s in text for s in ['© OpenStreetMap contributors', 'ohsome platform by HeiGIT', 'ODbL 1.0',
                                    'https://www.openstreetmap.org/copyright',
                                    'https://opendatacommons.org/licenses/odbl/1-0/', 'https://ohsome.org/copyrights'])
    before[notice] = sha(notice.read_bytes())
    assert all(sha(p.read_bytes()) == digest for p, digest in before.items())
    assert (ROOT/'results/e285_heigit_query_proposal.json').read_bytes() == plan_raw
    successes = sum(r['status'] == 'success' for r in result)
    return dict(experiment='E285', attempts=len(receipts), successes=successes,
                source_gate=all(r['status'] == 'success' for r in result), snapshots=result,
                retained_raw_bytes=raw_total, immutable_files={p.name: digest for p, digest in before.items()},
                independent_successful_snapshots_verified=successes, shared_native_parquet_parser=True,
                physical_route_or_publication_timing_verified=False, model_or_accuracy_evaluated=False)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--snapshot':
        print(json.dumps(snapshot(sys.stdin.buffer.read(2097153), json.loads(sys.argv[2]))))
    else:
        assert len(sys.argv) == 2, 'Read-only usage: research Python -m experiments.e285_heigit_audit SOURCE_DIRECTORY'
        print(json.dumps(audit(Path(sys.argv[1])), indent=2))
