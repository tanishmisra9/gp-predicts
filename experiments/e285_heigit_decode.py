"""E285 local-only Parquet decoder; its isolated process never receives a key."""
import json
import subprocess
import sys
from pathlib import Path

COLUMNS = ('osm_type', 'osm_id', 'version', 'minor_version', 'edit_timestamp',
           'geom_type', 'geom', 'clipped', 'tags')
ROOT = Path(__file__).resolve().parents[1]


def decode(raw):
    """Bounded bytes -> allowlisted records, verified native GEOMETRY CRS; fail closed."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    assert pa.__version__ == '26.0.1'
    assert isinstance(raw, bytes) and 12 <= len(raw) <= 2*1024**2
    assert raw[:4] == raw[-4:] == b'PAR1'
    with pq.ParquetFile(pa.BufferReader(raw), arrow_extensions_enabled=False,
                        pre_buffer=False) as source:
        metadata = source.metadata
        assert 0 < metadata.num_rows <= 10000
        schema = source.schema_arrow
        assert len(set(schema.names)) == len(schema.names)
        assert set(COLUMNS) <= set(schema.names)
        expected = dict(osm_type=pa.string(), osm_id=pa.int64(), version=pa.int32(),
                        minor_version=pa.int32(), edit_timestamp=pa.timestamp('us', tz='UTC'),
                        geom_type=pa.string(), geom=pa.binary(), clipped=pa.bool_(),
                        tags=pa.map_(pa.string(), pa.string()))
        assert all(schema.field(k).type == v for k, v in expected.items())
        selected = []; geom = []
        for i in range(len(source.schema)):
            column = source.schema.column(i)
            if column.path in COLUMNS or column.path.startswith('tags.'):
                selected.append(i)
            if column.path == 'geom':
                geom.append(column)
        assert len(geom) == 1 and geom[0].logical_type.type == 'GEOMETRY'
        logical = json.loads(geom[0].logical_type.to_json())
        assert logical['Type'] == 'Geometry'
        crs = logical.get('crs', 'OGC:CRS84')  # Native GEOMETRY default, never plain binary.
        assert crs in ('OGC:CRS84', 'EPSG:4326')
        declared = 0
        for g in range(metadata.num_row_groups):
            group = metadata.row_group(g)
            for i in selected:
                size = group.column(i).total_uncompressed_size
                assert size >= 0
                declared += size
        assert declared <= 16*1024**2
        table = source.read(columns=list(COLUMNS), use_threads=False, use_pandas_metadata=False)
        assert table.num_rows == metadata.num_rows and set(table.column_names) == set(COLUMNS)
        assert all(column.null_count == 0 for column in table.columns)
        rows = table.to_pylist()
    for row in rows:
        tags = row['tags']
        assert isinstance(tags, list) and all(isinstance(k, str) and isinstance(v, str) for k, v in tags)
        assert len({k for k, _ in tags}) == len(tags)
        highway = [v for k, v in tags if k == 'highway']
        assert highway == ['raceway']
        row['tags'] = {'highway': 'raceway'}
    return rows, crs


def decode_isolated(raw):
    """Research-model Python delegates only raw bytes to the pinned no-NumPy venv."""
    assert isinstance(raw, bytes) and 12 <= len(raw) <= 2*1024**2
    python = ROOT/'.venv-e285/bin/python'
    completed = subprocess.run([str(python), '-I', str(Path(__file__).resolve())],
                               input=raw, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               env={}, check=False)
    if completed.returncode:
        raise RuntimeError('E285 isolated Parquet decode failed')
    result = json.loads(completed.stdout)
    for row in result['records']:
        row['geom'] = bytes.fromhex(row['geom'])
    return result['records'], result['crs']


if __name__ == '__main__':
    records, crs = decode(sys.stdin.buffer.read(2*1024**2+1))
    for row in records:
        row['geom'] = row['geom'].hex()
        row['edit_timestamp'] = row['edit_timestamp'].isoformat()
    print(json.dumps(dict(records=records, crs=crs)))
