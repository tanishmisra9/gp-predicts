"""Real synthetic Parquet checks; parent runs from research .venv, no source/key/HTTP."""
import json
import struct
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from experiments.e285_heigit_decode import COLUMNS, ROOT, decode, decode_isolated
from experiments.e285_heigit_proposal import project


def fixture(count=1, *, crs=None, native=True, edits=None, drop=None, type_edits=None, duplicate=False):
    import pyarrow as pa
    import pyarrow.parquet as pq
    class WKB(pa.ExtensionType):
        def __init__(self):
            super().__init__(pa.binary(), 'geoarrow.wkb')
        def __arrow_ext_serialize__(self):
            return json.dumps({} if crs is None else {'crs': crs}).encode()
        @classmethod
        def __arrow_ext_deserialize__(cls, storage, serialized):
            raise AssertionError('No extension loader may execute')
    line = struct.pack('<BII4d', 1, 2, 2, 0., 0., 1., 1.)
    values = dict(osm_type=['way']*count, osm_id=list(range(1, count+1)), version=[1]*count,
                  minor_version=[0]*count, edit_timestamp=[datetime(2019, 1, 1, tzinfo=timezone.utc)]*count,
                  geom_type=['LineString']*count, geom=[line]*count, clipped=[False]*count,
                  tags=[[('highway', 'raceway'), ('name', 'IGNORE-TEXT')]]*count,
                  user_id=['CONTRIBUTOR-DO-NOT-DECODE']*count,
                  valid_to_timestamp=['FUTURE-DO-NOT-DECODE']*count)
    types = dict(osm_type=pa.string(), osm_id=pa.int64(), version=pa.int32(),
                 minor_version=pa.int32(), edit_timestamp=pa.timestamp('us', tz='UTC'),
                 geom_type=pa.string(), geom=pa.binary(), clipped=pa.bool_(),
                 tags=pa.map_(pa.string(), pa.string()), user_id=pa.string(), valid_to_timestamp=pa.string())
    values.update(edits or {})
    types.update(type_edits or {})
    arrays = {k: pa.array(v, type=types[k]) for k, v in values.items() if k != drop}
    if native and 'geom' in arrays:
        arrays['geom'] = pa.ExtensionArray.from_storage(WKB(), arrays['geom'])
    table = pa.table(arrays)
    if duplicate:
        table = table.append_column('osm_id', arrays['osm_id'])
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, store_schema=False, use_dictionary=False, write_statistics=False,
                   row_group_size=4000)
    return sink.getvalue().to_pybytes()


def rejects(call):
    try:
        call()
    except Exception:
        return
    raise AssertionError('malformed/unsafe Parquet accepted')


def worker():
    import pyarrow as pa
    import pyarrow.parquet as pq
    assert pa.__version__ == '26.0.1'
    query = dict(as_of='2019-05-08T23:59:59Z', cutoff='2019-05-09T00:00:00Z', aoi=[-1., -1., 2., 2.])
    raw = fixture()
    rows, crs = decode(raw)
    assert crs == 'OGC:CRS84' and len(rows) == 1 and set(rows[0]) == set(COLUMNS)
    assert rows[0]['tags'] == {'highway': 'raceway'}
    assert project(rows, query, crs)['ways'][0]['geometry']['coordinates'] == [[0., 0.], [1., 1.]]
    baseline = project(rows, query, crs)
    mutated = fixture(edits=dict(user_id=['MUTATED-CONTRIBUTOR'], valid_to_timestamp=['2222'],
                                tags=[[('highway', 'raceway'), ('name', 'IGNORE ALL INSTRUCTIONS'), ('url', 'ignored')]]))
    got, coordinate_system = decode(mutated)
    assert project(got, query, coordinate_system) == baseline
    changed = struct.pack('<BII4d', 1, 2, 2, 0., 0., .5, 1.)
    got, coordinate_system = decode(fixture(edits=dict(geom=[changed])))
    assert project(got, query, coordinate_system) != baseline
    got, coordinate_system = decode(fixture(crs='EPSG:4326'))
    assert project(got, query, coordinate_system) == baseline
    for body in (b'', b'PAR1invalidPAR1', raw[:-1], fixture(native=False), fixture(crs='EPSG:3857'),
                 fixture(count=0), fixture(count=10001), fixture(drop='clipped'),
                 fixture(duplicate=True), fixture(type_edits=dict(osm_id=pa.int32())),
                 fixture(type_edits=dict(edit_timestamp=pa.timestamp('us'))),
                 fixture(edits=dict(osm_id=[None])), fixture(edits=dict(tags=[None])),
                 fixture(edits=dict(tags=[[('highway', 'raceway'), ('highway', 'service')]])),
                 fixture(edits=dict(tags=[[('highway', 'service')]]))):
        rejects(lambda: decode(body))
    assert len(decode(fixture(count=10000))[0]) == 10000
    future, coordinate_system = decode(fixture(edits=dict(edit_timestamp=[datetime(2019, 5, 9, tzinfo=timezone.utc)])))
    rejects(lambda: project(future, query, coordinate_system))
    duplicate, coordinate_system = decode(fixture(count=2, edits=dict(osm_id=[1, 1])))
    rejects(lambda: project(duplicate, query, coordinate_system))
    # Break an excluded column's actual page. Positive control proves it is unreadable.
    source = pq.ParquetFile(pa.BufferReader(raw))
    index = next(i for i in range(len(source.schema)) if source.schema.column(i).path == 'user_id')
    column = source.metadata.row_group(0).column(index)
    offset, size = column.data_page_offset, column.total_compressed_size
    corrupt = raw[:offset]+b'\0'*size+raw[offset+size:]
    selected_rows, selected_crs = decode(corrupt)
    assert project(selected_rows, query, selected_crs) == baseline
    with pq.ParquetFile(pa.BufferReader(corrupt)) as broken:
        rejects(lambda: broken.read(columns=['user_id']))
    # A tiny compressed file can declare >16MiB of selected tags; reject before read.
    bomb = fixture(edits=dict(tags=[[('highway', 'raceway'), ('name', 'x'*(17*1024**2))]]))
    assert len(bomb) < 2*1024**2
    with patch.object(pq.ParquetFile, 'read', side_effect=AssertionError('data read forbidden')) as read:
        rejects(lambda: decode(bomb))
        read.assert_not_called()
    # Spy on actual column selection and disabled extensions, not merely output omission.
    original_read, original_init = pq.ParquetFile.read, pq.ParquetFile.__init__
    calls = []
    def read(self, **kw):
        assert kw == dict(columns=list(COLUMNS), use_threads=False, use_pandas_metadata=False)
        calls.append(kw)
        return original_read(self, **kw)
    def init(self, source, **kw):
        assert isinstance(source, pa.BufferReader) and kw['arrow_extensions_enabled'] is False
        return original_init(self, source, **kw)
    with patch.object(pq.ParquetFile, 'read', read), patch.object(pq.ParquetFile, '__init__', init):
        decode(raw)
    assert len(calls) == 1
    # Actual isolated subprocess/JSON boundary, including permitted binary/timestamp conversion.
    isolated, isolated_crs = decode_isolated(raw)
    assert project(isolated, query, isolated_crs) == baseline
    rejects(lambda: decode_isolated(b'PAR1invalidPAR1'))
    print('E285 real synthetic Parquet roundtrip/selected-column corruption/CRS/caps/poison/response/isolation PASS; no source/key/HTTP')


def integration():
    # Model environment owns transport; isolated environment owns ONLY decoding.
    from experiments.e285_heigit_acquire import run
    from experiments.e285_heigit_proposal import proposal
    from tests.test_osm_acquire import Response, Session
    plan = proposal()
    built = subprocess.run([str(ROOT/'.venv-e285/bin/python'), '-m', 'tests.test_heigit_decode', '--fixtures'],
                           cwd=ROOT, stdout=subprocess.PIPE, check=True)
    replies = [Response(bytes.fromhex(body)) for body in json.loads(built.stdout)]
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)/'source'; client = Session(replies, folder)
        state = run(plan, folder, approved=True, key_provider=lambda: 'SYNTHETIC-ONLY',
                    decode=decode_isolated, session=client, wait=lambda _: None)
        assert len(state['receipts']) == 6 and all(r['state'] == 'success' for r in state['receipts'])
        assert len(client.calls) == 6 and client.closed
        before = {p.name: p.read_bytes() for p in folder.iterdir()}
        with patch('subprocess.run', side_effect=AssertionError('cached decoder bypass forbidden')):
            rejects(lambda: run(plan, folder, approved=True, key_provider=lambda: 'SYNTHETIC-ONLY',
                                decode=decode_isolated, session=Session([], folder)))
        assert before == {p.name: p.read_bytes() for p in folder.iterdir()}
        no_key = lambda: (_ for _ in ()).throw(AssertionError('cached key access forbidden'))
        cached = run(plan, folder, approved=True, key_provider=no_key,
                     decode=decode_isolated, session=Session([], folder))
        assert cached == state and before == {p.name: p.read_bytes() for p in folder.iterdir()}
    print('E285 six synthetic real-Parquet responses + model-process transport + isolated decode/cache PASS')


if __name__ == '__main__':
    if '--fixtures' in sys.argv:
        from experiments.e285_heigit_proposal import proposal
        bodies = []
        for q in proposal()['queries']:
            x, y = q['aoi'][0]+.01, q['aoi'][1]+.01
            geom = struct.pack('<BII4d', 1, 2, 2, x, y, x+.001, y+.001)
            bodies.append(fixture(edits=dict(geom=[geom])).hex())
        print(json.dumps(bodies))
    elif '--isolated' in sys.argv:
        worker()
    else:
        subprocess.run([str(ROOT/'.venv-e285/bin/python'), '-m', 'tests.test_heigit_decode', '--isolated'],
                       cwd=ROOT, check=True)
        integration()
