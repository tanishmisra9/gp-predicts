"""E285 bounded runner, prepared with fake transport only; approval pending.

No CLI acquisition, real credential provider or Parquet decoder. Those last
two integrations remain gated on explicit scope approval. approved=True is an
operator assertion, never permission by itself. Tests inject dummy callbacks.
"""
import json
import time
from datetime import datetime, timezone

from experiments.e280_osm_acquire import digest, save
from experiments.e285_heigit_proposal import ROOT, ENDPOINT, project, proposal


def run(plan, directory, *, approved=False, key_provider=None, decode=None,
        session=None, wait=time.sleep):
    """Frozen plan + temporary source directory -> sticky, key-free receipts.

    decode(bytes) must return (selected records, verified file CRS). It is not
    supplied here: a fake decoder cannot establish actual Parquet validity.
    """
    assert approved is True, 'Explicit additional user approval required; goal continuation is not approval.'
    assert plan == proposal(), 'registered scope changed'
    assert callable(decode) and callable(key_provider)
    assert not directory.resolve().is_relative_to(ROOT)
    limit = plan['limits']['max_body_bytes']
    identity = digest(json.dumps(plan, sort_keys=True).encode())
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory/'acquisition.lock'
    with lock.open('x') as file:
        file.write('E285 active; inspect authoritative process before removing any stale lock\n')
    state_path = directory/'receipts.json'
    try:
        assert not state_path.is_symlink()
        state = (json.loads(state_path.read_text()) if state_path.exists()
                 else dict(proposal_sha256=identity, receipts=[]))
        assert state['proposal_sha256'] == identity
        reports = state['receipts']; queries = plan['queries']
        assert isinstance(reports, list) and len(reports) <= 6
        expected = set()
        for i, receipt in enumerate(reports):
            q = queries[i]
            assert receipt['index'] == i and receipt['request_sha256'] == q['request_sha256']
            assert receipt['state'] in ('started', 'success', 'failed')
            if receipt['state'] != 'success':
                assert i == len(reports)-1
                return state  # An interrupted/failed attempt is never retried.
            raw_name = q['request_sha256']+'.parquet'
            projection_name = q['request_sha256']+'.projection.json'
            assert receipt['raw_file'] == raw_name and receipt['projection_file'] == projection_name
            expected.update((raw_name, projection_name))
            raw_path, projection_path = directory/raw_name, directory/projection_name
            assert not raw_path.is_symlink() and not projection_path.is_symlink()
            try:
                raw = raw_path.read_bytes()
                assert len(raw) == receipt['bytes'] <= limit and digest(raw) == receipt['raw_sha256']
                records, crs = decode(raw)
                projection = project(records, q, crs)
                saved = projection_path.read_bytes()
                assert digest(saved) == receipt['projection_sha256']
                assert json.loads(saved) == projection
            except Exception as exc:
                raise RuntimeError('E285 cached source invalid: '+type(exc).__name__) from None
        present = {p.name for pattern in ('*.parquet', '*.projection.json') for p in directory.glob(pattern)}
        assert present == expected, 'orphan source files; inspect receipts, do not refetch'
        if len(reports) == 6:
            return state
        if session is None:
            import requests
            session = requests.Session()
            session.trust_env = False
            session.mount('https://', requests.adapters.HTTPAdapter(max_retries=0))
        try:
            notice = ('© OpenStreetMap contributors; ohsome platform by HeiGIT — ODbL 1.0\n'
                      'https://www.openstreetmap.org/copyright\n'
                      'https://opendatacommons.org/licenses/odbl/1-0/\n'
                      'https://ohsome.org/copyrights\n'
                      'E285 six historical snapshots; exact scope in registered request manifest.\n'
                      'Raw files temporary, pending independent audit then deletion. Contributor columns not decoded.\n'
                      'Projection omits text/identity/future-validity fields; no certified physical route or publication date.\n')
            notice_path = directory/'SOURCE_NOTICE.txt'
            assert not notice_path.is_symlink()
            if notice_path.exists():
                assert notice_path.read_text() == notice
            else:
                with notice_path.open('x') as file:
                    file.write(notice)
            try:
                token = key_provider()
                assert isinstance(token, str) and token and token.strip() == token
                assert '\r' not in token and '\n' not in token
            except Exception as exc:
                raise RuntimeError('E285 credential setup failed: '+type(exc).__name__) from None
            for i in range(len(reports), 6):
                if reports:
                    wait(plan['limits']['start_spacing_seconds'])
                q = queries[i]
                assert q['request']['endpoint'] == ENDPOINT and q['request']['method'] == 'GET'
                receipt = dict(index=i, request_sha256=q['request_sha256'], state='started',
                               access_started_utc=datetime.now(timezone.utc).isoformat())
                reports.append(receipt)
                save(state_path, state)  # Reserve attempt before entering transport.
                try:
                    with session.get(ENDPOINT, params=q['request']['params'], stream=True,
                                     allow_redirects=False, timeout=(10, 35),
                                     headers={'Accept': 'application/octet-stream',
                                              'Authorization': token,
                                              'User-Agent': 'gp-predicts-research/E285'}) as response:
                        receipt['status_code'] = int(response.status_code)
                        assert response.status_code == 200, 'non-200 source stop'
                        length = response.headers.get('Content-Length')
                        if length is not None:
                            assert 0 <= int(length) <= limit, 'declared body limit'
                        body = bytearray()
                        for chunk in response.iter_content(chunk_size=4096):
                            assert len(body)+len(chunk) <= limit, 'streamed body limit'
                            body.extend(chunk)
                    raw = bytes(body)
                    assert len(raw) >= 12 and raw[:4] == raw[-4:] == b'PAR1'
                    raw_name = q['request_sha256']+'.parquet'
                    with (directory/raw_name).open('xb') as file:
                        file.write(raw)
                    receipt.update(raw_file=raw_name, bytes=len(raw), raw_sha256=digest(raw))
                    records, crs = decode(raw)
                    projection = project(records, q, crs)
                    projection_name = q['request_sha256']+'.projection.json'
                    save(directory/projection_name, projection)
                    saved = (directory/projection_name).read_bytes()
                    receipt.update(state='success', projection_file=projection_name,
                                   projection_sha256=digest(saved), ways=len(projection['ways']))
                except Exception as exc:
                    # Exception messages/response bodies can echo Authorization.
                    receipt.update(state='failed', error_type=type(exc).__name__)
                    save(state_path, state)
                    print('E285', i, 'failed', receipt.get('status_code'), receipt['error_type'], flush=True)
                    break
                save(state_path, state)
                print('E285', i, 'success', receipt['status_code'], receipt['bytes'], flush=True)
        finally:
            session.close()
        return state
    finally:
        lock.unlink()


if __name__ == '__main__':
    print('E285 offline runner only. Additional approval, real decoder and credential integration still required.')
