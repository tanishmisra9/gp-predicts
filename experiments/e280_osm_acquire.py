"""E280 bounded transport, prepared offline. Actual ODbL approval is still pending.

No command-line acquisition entry point. The operator must have the user's
explicit approval before calling run(approved=True); the flag is not permission.
"""
import hashlib
import json
import time
from datetime import datetime, timezone

from experiments.e280_osm_proposal import ENDPOINT, manifest, project

LIMIT = 2*1024**2


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, state):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2)+'\n'); temporary.replace(path)


def run(proposal, directory, *, approved=False, session=None, wait=time.sleep):
    assert approved is True, 'Explicit user ODbL approval required; automatic goal continuation is not approval.'
    assert proposal == manifest(), 'registered six-query scope or source hashes changed'
    identity = digest(json.dumps(proposal, sort_keys=True).encode())
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / 'acquisition.lock'
    # A stale lock requires process/receipt review, never automatic removal/retry.
    with lock.open('x') as file: file.write('E280 acquisition active; inspect process before removal\n')
    state_path = directory / 'receipts.json'
    try:
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(proposal_sha256=identity, receipts=[])
        assert state['proposal_sha256'] == identity
        reports = state['receipts']; queries = proposal['queries']
        assert len(reports) <= 6
        for i, receipt in enumerate(reports):
            assert receipt['index'] == i and receipt['query_sha256'] == queries[i]['query_sha256']
            assert receipt['state'] in ['started', 'success', 'failed']
            if receipt['state'] != 'success':
                assert i == len(reports)-1
                return state  # Interrupted/failed attempt is sticky; no automatic retry.
            raw = (directory / receipt['raw_file']).read_bytes()
            assert len(raw) == receipt['bytes'] <= LIMIT and digest(raw) == receipt['raw_sha256']
            projection = project(json.loads(raw.decode('utf-8')), queries[i]['as_of'], queries[i]['cutoff'])
            assert digest(json.dumps(projection, sort_keys=True).encode()) == receipt['projection_sha256']
        if len(reports) == 6: return state
        if session is None:
            import requests
            session = requests.Session(); session.trust_env = False
            session.mount('https://', requests.adapters.HTTPAdapter(max_retries=0))
        try:
            (directory / 'SOURCE_NOTICE.txt').write_text(
                '© OpenStreetMap contributors — ODbL 1.0\nhttps://www.openstreetmap.org/copyright\n'
                'https://opendatacommons.org/licenses/odbl/1-0/\n'
                'E280 six historical snapshots; exact query/cutoff in registered proposal.\n'
                'Raw JSON retained for audit; projection removes contributor identities and free text.\n'
                'Mapping state is not certified physical Grand Prix geometry.\n')
            for i in range(len(reports), 6):
                if reports: wait(2)
                q = queries[i]; assert q['endpoint'] == ENDPOINT
                receipt = dict(index=i, query_sha256=q['query_sha256'], state='started',
                               access_started_utc=datetime.now(timezone.utc).isoformat())
                reports.append(receipt); save(state_path, state)  # Reserve attempt BEFORE transport.
                try:
                    with session.get(ENDPOINT, params={'data': q['query']}, stream=True, allow_redirects=False, timeout=35,
                                     headers={'Accept': 'application/json', 'User-Agent': 'gp-predicts-research/E280'}) as response:
                        receipt['status_code'] = int(response.status_code)
                        assert response.status_code == 200, 'non-200 source stop'
                        length = response.headers.get('Content-Length')
                        if length is not None: assert 0 <= int(length) <= LIMIT, 'declared body limit'
                        body = bytearray()
                        for chunk in response.iter_content(chunk_size=4096):
                            assert len(body)+len(chunk) <= LIMIT, 'streamed body limit'
                            body.extend(chunk)
                    raw = bytes(body)
                    filename = q['query_sha256']+'.json'
                    with (directory / filename).open('xb') as file: file.write(raw)
                    receipt.update(raw_file=filename, bytes=len(raw), raw_sha256=digest(raw))
                    projection = project(json.loads(raw.decode('utf-8')), q['as_of'], q['cutoff'])
                    receipt.update(state='success', nodes=len(projection['nodes']), ways=len(projection['ways']),
                                   projection_sha256=digest(json.dumps(projection, sort_keys=True).encode()))
                except Exception as exc:
                    receipt.update(state='failed', error_type=type(exc).__name__)
                    save(state_path, state)
                    print('E280', i, receipt['state'], receipt.get('status_code'), receipt['error_type'], flush=True)
                    break
                save(state_path, state)
                print('E280', i, receipt['state'], receipt['status_code'], receipt['bytes'], flush=True)
        finally:
            session.close()
        return state
    finally:
        lock.unlink()


if __name__ == '__main__':
    print('E280 runner prepared only. No acquisition: explicit ODbL approval is pending.')
