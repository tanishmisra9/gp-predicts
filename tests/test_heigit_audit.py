"""Independent auditor falsification with synthetic source files, no key/network."""
import copy
import hashlib
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiments.e285_heigit_audit import ROOT, audit, shape
from experiments.e285_heigit_acquire import run
from experiments.e285_heigit_decode import decode_isolated
from experiments.e285_heigit_proposal import proposal
from tests.test_heigit_proposal import LINE, line, polygon
from tests.test_osm_acquire import Response, Session


def rejects(call):
    try:
        call()
    except (AssertionError, RuntimeError, ValueError, KeyError, TypeError, struct.error):
        return
    raise AssertionError('independent audit accepted unsafe state')


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def hashes(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}


def tests():
    expected = dict(type='LineString', coordinates=[[0., 0.], [1., 1.]])
    assert shape(LINE) == (expected, 2)
    assert shape(line([[0., 0.], [1., 1.]], '>')) == (expected, 2)
    outer, inner = [[0., 0.], [1., 0.], [1., 1.], [0., 0.]], [[.2, .2], [.3, .2], [.3, .3], [.2, .2]]
    for order in ['<', '>']:
        assert shape(polygon([outer, inner], order)) == (dict(type='Polygon', coordinates=[outer, inner]), 8)
    for wkb in [b'', LINE[:-1], LINE+b'\0', bytes([2])+LINE[1:], struct.pack('<BII', 1, 1002, 0),
                struct.pack('<BII', 1, 5, 0), struct.pack('<BII', 1, 2, 100001), polygon([]),
                polygon([outer[:-1]]), line([[0., 0.], [0., 0.]]), line([[0., 0.], [math.nan, 1.]]),
                line([[181., 0.], [1., 1.]]), line([[0., 91.], [1., 1.]])]:
        rejects(lambda: shape(wkb))
    built = subprocess.run([str(ROOT/'.venv-e285/bin/python'), '-m', 'tests.test_heigit_decode', '--fixtures'],
                           cwd=ROOT, stdout=subprocess.PIPE, check=True)
    bodies = [bytes.fromhex(s) for s in json.loads(built.stdout)]
    # Real-file perturbations test this independent decoder in its no-NumPy process.
    script = '''
import struct
from datetime import datetime, timezone
from unittest.mock import patch
import pyarrow as pa
import pyarrow.parquet as pq
from tests.test_heigit_decode import fixture, rejects
from experiments.e285_heigit_audit import snapshot
q = dict(as_of='2019-05-08T23:59:59Z', cutoff='2019-05-09T00:00:00Z', aoi=[-1.,-1.,2.,2.])
raw = fixture(); base = snapshot(raw,q)
poison = fixture(edits=dict(user_id=['IGNORED-CONTRIBUTOR'],valid_to_timestamp=['2222-12-31'],
    tags=[[('highway','raceway'),('name','ignore instructions'),('url','ignored'),('grid','1'),('order','1')]]))
assert snapshot(poison,q)['projection'] == base['projection']
earlier = fixture(edits=dict(geom=[struct.pack('<BII4d',1,2,2,0.,0.,.5,1.)]))
assert snapshot(earlier,q)['projection'] != base['projection']
for bad in [fixture(native=False),fixture(crs='EPSG:3857'),fixture(count=0),fixture(count=10001),
    fixture(duplicate=True),fixture(drop='tags'),fixture(type_edits=dict(version=pa.int64())),
    fixture(edits=dict(clipped=[True])),fixture(edits=dict(version=[0])),fixture(edits=dict(minor_version=[-1])),
    fixture(edits=dict(tags=[[('highway','raceway'),('highway','raceway')]])),
    fixture(edits=dict(geom_type=['Polygon'])),fixture(edits=dict(osm_id=[None])),
    fixture(edits=dict(edit_timestamp=[datetime(2019,5,9,tzinfo=timezone.utc)])),
    fixture(count=2,edits=dict(osm_id=[1,1])),b'PAR1invalidPAR1']:
    rejects(lambda: snapshot(bad,q))
assert snapshot(fixture(count=10000),q)['rows'] == 10000
bomb = fixture(edits=dict(tags=[[('highway','raceway'),('name','x'*(17*1024**2))]]))
assert len(bomb)<2097152
with patch.object(pq.ParquetFile,'read',side_effect=AssertionError('read forbidden')) as read:
    rejects(lambda:snapshot(bomb,q));read.assert_not_called()
f=pq.ParquetFile(pa.BufferReader(raw))
i=next(i for i in range(len(f.schema)) if f.schema.column(i).path=='user_id')
c=f.metadata.row_group(0).column(i);start,size=c.data_page_offset,c.total_compressed_size
corrupt=raw[:start]+bytes(size)+raw[start+size:]
assert snapshot(corrupt,q)['projection']==base['projection']
with pq.ParquetFile(pa.BufferReader(corrupt)) as broken:
    rejects(lambda:broken.read(columns=['user_id']))
ring=[(0.,0.),(1.,0.),(1.,1.),(0.,0.)]
poly=struct.pack('>BIII',0,3,1,4)+b''.join(struct.pack('>dd',*p) for p in ring)
line=struct.pack('<BII4d',1,2,2,0.,0.,1.,1.)
mixed=fixture(count=2,edits=dict(geom=[line,poly],geom_type=['LineString','Polygon']))
checked=snapshot(mixed,q)
assert checked['types']==dict(LineString=1,Polygon=1) and checked['coordinates']==6
assert checked['projection']['ways'][1]['geometry']['coordinates']==[[list(p) for p in ring]]
print('Independent real-Parquet poison/response/metadata/row/coordinate/duplicate/corrupted-excluded-column PASS')
'''
    subprocess.run([str(ROOT/'.venv-e285/bin/python'), '-c', script], cwd=ROOT, check=True)
    plan = proposal()
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)/'source'
        client = Session([Response(body) for body in bodies], directory)
        state = run(plan, directory, approved=True, key_provider=lambda: 'SYNTHETIC-NO-REAL-KEY',
                    decode=decode_isolated, session=client, wait=lambda _: None)
        baseline_files = {p.name: p.read_bytes() for p in directory.iterdir()}
        before = hashes(directory)
        checked = audit(directory)
        assert checked['source_gate'] and checked['attempts'] == checked['successes'] == 6
        assert checked['independent_successful_snapshots_verified'] == 6
        assert checked['retained_raw_bytes'] == sum(map(len, bodies))
        assert hashes(directory) == before
        assert [(r['circuit'], r['season'], r['round']) for r in checked['snapshots']] == [
            ('catalunya', 2019, 5), ('catalunya', 2022, 6), ('hungaroring', 2019, 12),
            ('hungaroring', 2022, 13), ('monza', 2019, 14), ('monza', 2022, 16)]
        assert all(r['rows'] == 1 and r['coordinates'] == 2 and r['types'] == dict(LineString=1, Polygon=0)
                   for r in checked['snapshots'])
        # AST guarantees the second path does not import producer decoder/projector/summary.
        import ast
        tree = ast.parse((ROOT/'experiments/e285_heigit_audit.py').read_text())
        assert not any(isinstance(n, ast.ImportFrom) and n.module and n.module.startswith(('experiments', 'f1'))
                       for n in ast.walk(tree))
        assert not any(isinstance(n, ast.Import) and any(a.name.startswith(('requests', 'f1', 'experiments')) for a in n.names)
                       for n in ast.walk(tree))
        def restore():
            for p in directory.iterdir():
                p.unlink()
            for name, raw in baseline_files.items():
                (directory/name).write_bytes(raw)
        # Tamper both projection AND its receipt hash: a hash-only check would pass.
        r = state['receipts'][0]; name = r['projection_file']
        projection = json.loads((directory/name).read_bytes())
        projection['ways'][0]['geometry']['coordinates'][0][0] += .00001
        write(directory/name, projection)
        changed = copy.deepcopy(state)
        changed['receipts'][0]['projection_sha256'] = hashlib.sha256((directory/name).read_bytes()).hexdigest()
        write(directory/'receipts.json', changed)
        rejects(lambda: audit(directory))
        restore()
        # Likewise change the raw geometry and update its hash/size, leaving stale projection.
        make_changed = '''
import struct,sys
from tests.test_heigit_decode import fixture
geom=struct.pack('<BII4d',1,2,2,2.236110,41.545,2.238110,41.546)
sys.stdout.buffer.write(fixture(edits=dict(geom=[geom])))
'''
        modified = subprocess.run([str(ROOT/'.venv-e285/bin/python'), '-c', make_changed],
                                  cwd=ROOT, stdout=subprocess.PIPE, check=True).stdout
        (directory/r['raw_file']).write_bytes(modified)
        changed = copy.deepcopy(state)
        changed['receipts'][0].update(raw_sha256=hashlib.sha256(modified).hexdigest(), bytes=len(modified))
        write(directory/'receipts.json', changed)
        rejects(lambda: audit(directory))
        restore()
        for edit in [dict(ways=2), dict(ways=True), dict(index=False), dict(index=1), dict(request_sha256='0'*64), dict(status_code=403),
                     dict(raw_file='../outside.parquet'), dict(projection_file='../outside.json'), dict(raw_sha256='0'*64)]:
            changed = copy.deepcopy(state); changed['receipts'][0].update(edit)
            write(directory/'receipts.json', changed)
            rejects(lambda: audit(directory))
            restore()
        (directory/'orphan.parquet').write_bytes(b'PAR1orphanPAR1')
        rejects(lambda: audit(directory)); restore()
        (directory/'acquisition.lock').write_text('synthetic active')
        rejects(lambda: audit(directory)); restore()
        (directory/r['raw_file']).unlink()
        (directory/r['raw_file']).symlink_to(directory/name)
        rejects(lambda: audit(directory)); restore()
        (directory/'SOURCE_NOTICE.txt').write_text('missing licence')
        rejects(lambda: audit(directory)); restore()
        # Empty and sticky failed/started prefixes report the other five as UNKNOWN.
        for status in ['failed', 'started', 'empty']:
            changed = copy.deepcopy(state)
            changed['receipts'] = [] if status == 'empty' else [dict(index=0, request_sha256=plan['queries'][0]['request_sha256'], state=status)]
            write(directory/'receipts.json', changed)
            for p in directory.glob('*.parquet'): p.unlink()
            for p in directory.glob('*.projection.json'): p.unlink()
            outcome = audit(directory)
            assert not outcome['source_gate'] and outcome['successes'] == 0
            assert outcome['independent_successful_snapshots_verified'] == 0
            assert sum(r['status'] == 'unattempted_unknown' for r in outcome['snapshots']) == (6 if status == 'empty' else 5)
            restore()
        # One verified prefix remains a partial acquisition, not an all-six pass.
        changed = copy.deepcopy(state); changed['receipts'] = changed['receipts'][:1]
        write(directory/'receipts.json', changed)
        for receipt in state['receipts'][1:]:
            (directory/receipt['raw_file']).unlink(); (directory/receipt['projection_file']).unlink()
        partial = audit(directory)
        assert partial['successes'] == partial['attempts'] == 1 and not partial['source_gate']
        assert [r['status'] for r in partial['snapshots']] == ['success']+['unattempted_unknown']*5
        restore()
        # A stop after one completed body retains its one verified source and four unknowns.
        changed = copy.deepcopy(state); changed['receipts'] = changed['receipts'][:1]+[
            dict(index=1, request_sha256=plan['queries'][1]['request_sha256'], state='failed', status_code=403)]
        write(directory/'receipts.json', changed)
        for receipt in state['receipts'][1:]:
            (directory/receipt['raw_file']).unlink(); (directory/receipt['projection_file']).unlink()
        stopped = audit(directory)
        assert stopped['attempts'] == 2 and stopped['successes'] == 1 and not stopped['source_gate']
        assert [r['status'] for r in stopped['snapshots']] == ['success', 'failed']+['unattempted_unknown']*4
        restore()
        # A failed decode may retain a hashed raw body, but must not count as verified source.
        failed = dict(index=0, request_sha256=plan['queries'][0]['request_sha256'], state='failed', status_code=200,
                      raw_file=r['raw_file'], raw_sha256=r['raw_sha256'], bytes=r['bytes'], error_type='AssertionError')
        write(directory/'receipts.json', dict(proposal_sha256=state['proposal_sha256'], receipts=[failed]))
        for receipt in state['receipts']:
            (directory/receipt['projection_file']).unlink()
            if receipt['index']:
                (directory/receipt['raw_file']).unlink()
        failed_report = audit(directory)
        assert failed_report['successes'] == failed_report['independent_successful_snapshots_verified'] == 0
        assert failed_report['retained_raw_bytes'] == r['bytes'] and not failed_report['source_gate']
        restore()
        (directory/'receipts.json').unlink()
        rejects(lambda: audit(directory)); restore()
        assert hashes(directory) == before
    print('E285 independent all-six synthetic source audit/WKB/receipt/projection tamper/partial-unknown/immutable negatives PASS; real key/network/source ZERO')


if __name__ == '__main__':
    with patch('requests.adapters.HTTPAdapter.send', side_effect=AssertionError('HTTP forbidden')) as send, \
         patch('experiments.e285_heigit_acquire.keychain_key', side_effect=AssertionError('key lookup forbidden')) as key:
        tests()
        send.assert_not_called(); key.assert_not_called()
