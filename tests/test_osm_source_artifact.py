"""E280 independent raw transport/projection audit after authorized acquisition."""
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from experiments.e280_osm_proposal import manifest, project
from f1.data import ROOT

RAW = Path('/tmp/gp-predicts-e280-osm-source')


def independent(body, q):
    assert isinstance(body, dict) and not body.get('remark')
    elements = body['elements']; assert isinstance(elements, list) and 0 < len(elements) <= 100000
    boundary = datetime.fromisoformat(q['cutoff'].replace('Z', '+00:00'))
    limit = datetime.fromisoformat(q['as_of'].replace('Z', '+00:00'))
    output = {'nodes': [], 'ways': []}; keys = set(); node_ids = set(); references = set()
    for e in elements:
        kind, ident, version = e['type'], e['id'], e['version']
        assert kind in ['node', 'way'] and type(ident) is int and ident > 0 and type(version) is int and version > 0
        assert (kind, ident) not in keys; keys.add((kind, ident))
        stamp = datetime.fromisoformat(e['timestamp'].replace('Z', '+00:00'))
        assert e['timestamp'] == stamp.strftime('%Y-%m-%dT%H:%M:%SZ') and stamp.tzinfo == timezone.utc and stamp <= limit < boundary
        if kind == 'node':
            assert type(e['lat']) in [int, float] and type(e['lon']) in [int, float]
            assert math.isfinite(e['lat']) and math.isfinite(e['lon']) and -90 <= e['lat'] <= 90 and -180 <= e['lon'] <= 180
            node_ids.add(ident)
            output['nodes'].append({k:e[k] for k in ['id', 'version', 'timestamp', 'lat', 'lon']})
        else:
            assert e['tags']['highway'] == 'raceway' and isinstance(e['nodes'], list) and len(set(e['nodes'])) >= 2
            assert all(type(n) is int and n > 0 for n in e['nodes'])
            references.update(e['nodes'])
            output['ways'].append(dict(id=ident, version=version, timestamp=e['timestamp'], nodes=e['nodes'], highway='raceway'))
    assert output['ways'] and len(node_ids) >= 2 and references == node_ids
    for values in output.values(): values.sort(key=lambda r:r['id'])
    return output


def main():
    proposal = manifest(); qlist = proposal['queries']
    state = json.loads((RAW/'receipts.json').read_text())
    before = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RAW.iterdir() if p.is_file()}
    assert not (RAW/'acquisition.lock').exists()
    assert state['proposal_sha256'] == hashlib.sha256(json.dumps(proposal, sort_keys=True).encode()).hexdigest()
    actual = json.loads((RAW/'transport_audit.json').read_text())
    receipts = state['receipts']; assert len(receipts) <= 6 and len(actual) == len(receipts)
    projected = []; counts = []; total = 0; poisoned = 0
    for i, q in enumerate(qlist):
        if i >= len(receipts):
            counts.append(dict(index=i, circuit=q['circuit'], season=q['season'], state='unattempted', available=None)); continue
        r = receipts[i]; transport = actual[i]
        assert r['index'] == i and r['query_sha256'] == q['query_sha256'] and r['state'] in ['success', 'failed']
        parts = urlsplit(transport['url'])
        assert parts.scheme == 'https' and parts.netloc == 'overpass-api.de' and parts.path == '/api/interpreter'
        assert not parts.fragment and parse_qs(parts.query) == {'data':[q['query']]} and transport['method'] == 'GET'
        assert transport['status_code'] == r.get('status_code')
        row = dict(index=i, circuit=q['circuit'], season=q['season'], state=r['state'], available=r['state']=='success')
        row.update(status_code=r.get('status_code'), error_type=r.get('error_type'))
        if r['state'] == 'failed': assert i == len(receipts)-1
        if 'raw_file' in r:
            raw = (RAW/r['raw_file']).read_bytes()
            assert r['raw_file'] == q['query_sha256']+'.json' and len(raw) == r['bytes'] <= 2*1024**2
            assert hashlib.sha256(raw).hexdigest() == r['raw_sha256']; total += len(raw)
        if r['state'] == 'success':
            assert r['status_code'] == 200
            body = json.loads(raw.decode('utf-8')); p = independent(body, q)
            assert p == project(body, q['as_of'], q['cutoff'])
            assert hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest() == r['projection_sha256']
            assert len(p['nodes']) == r['nodes'] and len(p['ways']) == r['ways']
            discard = copy.deepcopy(body)
            for e in discard['elements']:
                e.update(user='poison', uid=-1, changeset=-1, note='poison', url='poison')
                e.setdefault('tags', {}).update(name='poison', note='poison', description='poison')
            assert project(discard, q['as_of'], q['cutoff']) == p
            prior = copy.deepcopy(body); node = next(e for e in prior['elements'] if e['type'] == 'node')
            node['lat'] += 1e-8 if node['lat'] < 89 else -1e-8
            assert project(prior, q['as_of'], q['cutoff']) != p
            for kind in ['node', 'way']:
                future = copy.deepcopy(body); next(e for e in future['elements'] if e['type'] == kind)['timestamp'] = q['cutoff']
                try: project(future, q['as_of'], q['cutoff'])
                except AssertionError: pass
                else: raise AssertionError('future raw element accepted')
            poisoned += 1; row.update(nodes=r['nodes'], ways=r['ways'], bytes=r['bytes'])
            projected.append(dict(index=i, circuit=q['circuit'], season=q['season'], as_of=q['as_of'], cutoff=q['cutoff'], **p))
        counts.append(row)
    assert total <= 12*1024**2
    assert before == {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RAW.iterdir() if p.is_file()}
    gate = len(receipts) == 6 and all(r['state'] == 'success' for r in receipts)
    result = dict(source_gate=gate, query_count=6, attempted=len(actual), available=len(projected), unattempted=6-len(actual),
                  accepted_raw_json_bytes=total, all_snapshot_counts=counts, independently_projected=len(projected),
                  request_receipts=receipts, transport_receipts=actual,
                  actual_raw_poison_response_checks=poisoned, source_immutable=True, raw_hashes=before,
                  code_hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
                               ['experiments/e280_osm_proposal.py','experiments/e280_osm_acquire.py','tests/test_osm_source_artifact.py']},
                  approval='User: Approve the six-snapshot OSM pilot', attribution='© OpenStreetMap contributors',
                  licence='https://opendatacommons.org/licenses/odbl/1-0/', snapshot_semantics='historical mapped state, not certified physical GP layout',
                  winner_model_fitted=False, outer_evaluated=False, evaluated_2026=False)
    projection_path = ROOT/'results/e280_osm_projection.json'
    projection_path.write_text(json.dumps(dict(attribution=result['attribution'], licence=result['licence'], snapshots=projected), indent=2)+'\n')
    result['projection_sha256'] = hashlib.sha256(projection_path.read_bytes()).hexdigest()
    (ROOT/'results/e280_source_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print('E280 independent actual transport/all-element projection/timestamps/references/poison/response/cache audit:',json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
