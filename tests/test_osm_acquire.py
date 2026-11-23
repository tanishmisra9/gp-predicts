"""E280 fake transport only; real HTTP forbidden throughout all scenarios."""
import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiments.e280_osm_acquire import LIMIT, run
from experiments.e280_osm_proposal import ENDPOINT, manifest


def source(q):
    return dict(elements=[dict(type='node', id=n, version=1, timestamp=q['as_of'], lat=40.+n/100, lon=2.) for n in [1, 2]]+
                         [dict(type='way', id=1, version=1, timestamp=q['as_of'], nodes=[1, 2], tags=dict(highway='raceway'))])


class Response:
    def __init__(self, payload, status=200, headers=None, chunks=None, stream_error=False):
        self.payload = payload; self.status_code = status; self.headers = headers or {}; self.closed = False
        self.chunks = chunks; self.reads = 0; self.stream_error = stream_error
    def __enter__(self): return self
    def __exit__(self, *args): self.closed = True
    def iter_content(self, chunk_size):
        assert chunk_size == 4096
        if self.stream_error: raise OSError('synthetic stream failure')
        for value in self.chunks if self.chunks is not None else [self.payload]:
            self.reads += 1; yield value


class Session:
    def __init__(self, responses, directory, fail=None):
        self.responses = responses; self.directory = directory; self.calls = []; self.closed = False; self.fail = fail
    def get(self, url, **kwargs):
        state = json.loads((self.directory / 'receipts.json').read_text())
        assert state['receipts'][-1]['state'] == 'started'  # On disk BEFORE request.
        i = len(self.calls); self.calls.append((url, kwargs))
        if self.fail is not None: raise self.fail
        return self.responses[i]
    def close(self): self.closed = True


def tests():
    proposal = manifest()
    def responses(): return [Response(json.dumps(source(q)).encode()) for q in proposal['queries']]
    with tempfile.TemporaryDirectory() as folder:
        parent = Path(folder)
        # Lack of approval and modified scopes fail before any transport/directory.
        directory = parent/'not-approved'; client = Session([], directory)
        for changed, approved in [(proposal, False), (dict(proposal, licence='changed'), True)]:
            try: run(changed, directory, approved=approved, session=client)
            except AssertionError: pass
            else: raise AssertionError('authorization/scope gate missing')
        assert not directory.exists() and not client.calls
        directory = parent/'success'; replies = responses(); client = Session(replies, directory); gaps = []
        state = run(proposal, directory, approved=True, session=client, wait=gaps.append)
        assert len(client.calls) == 6 and gaps == [2]*5 and client.closed and all(r.closed for r in replies)
        assert all(r['state'] == 'success' and r['nodes'] == 2 and r['ways'] == 1 for r in state['receipts'])
        for (_, kwargs), q in zip(client.calls, proposal['queries']):
            assert kwargs == dict(params={'data': q['query']}, stream=True, allow_redirects=False, timeout=35,
                                  headers={'Accept': 'application/json', 'User-Agent': 'gp-predicts-research/E280'})
        assert all(url == ENDPOINT for url, kw in client.calls)
        fresh = Session([], directory)
        assert run(proposal, directory, approved=True, session=fresh, wait=lambda _: None) == state and not fresh.calls
        # A clean completed prefix resumes ONLY unattempted keys, preserving original raw files.
        before = {p.name: p.read_bytes() for p in directory.glob('*.json') if p.name != 'receipts.json'}
        partial = copy.deepcopy(state); partial['receipts'] = partial['receipts'][:2]
        for r in state['receipts'][2:]: (directory/r['raw_file']).unlink()
        (directory/'receipts.json').write_text(json.dumps(partial))
        resumed = Session(responses()[2:], directory); waits = []
        done = run(proposal, directory, approved=True, session=resumed, wait=waits.append)
        assert len(done['receipts']) == 6 and len(resumed.calls) == 4 and waits == [2]*4
        for r in done['receipts'][:2]: assert (directory/r['raw_file']).read_bytes() == before[r['raw_file']]
        # Simulated crash checkpoint cannot be retried. Active/stale locks also block.
        interrupted = copy.deepcopy(done); interrupted['receipts'] = interrupted['receipts'][:3]; interrupted['receipts'][-1] = dict(index=2, state='started', query_sha256=proposal['queries'][2]['query_sha256'])
        (directory/'receipts.json').write_text(json.dumps(interrupted)); untouched = Session([], directory)
        assert run(proposal, directory, approved=True, session=untouched) == interrupted and not untouched.calls
        (directory/'acquisition.lock').write_text('synthetic live owner')
        try: run(proposal, directory, approved=True, session=untouched)
        except FileExistsError: pass
        else: raise AssertionError('concurrent invocation accepted')
        assert not untouched.calls; (directory/'acquisition.lock').unlink()
        # Inclusive retained-body boundary; cached bytes cannot be silently repaired/refetched.
        d = parent/'exact-limit'; replies = responses()
        replies[0].payload += b' '*(LIMIT-len(replies[0].payload))
        c = Session(replies, d); exact = run(proposal, d, approved=True, session=c, wait=lambda _: None)
        assert len(c.calls) == 6 and exact['receipts'][0]['bytes'] == LIMIT
        path = d/exact['receipts'][0]['raw_file']; original = path.read_bytes(); path.write_bytes(original+b' ')
        no_send = Session([], d)
        try: run(proposal, d, approved=True, session=no_send)
        except AssertionError: pass
        else: raise AssertionError('corrupt source cache accepted')
        assert not no_send.calls and not (d/'acquisition.lock').exists()
        # Every failure is sticky, closes the stream/session, and prevents all later sends.
        broken = [Response(b'', status=status) for status in [301, 401, 403, 429, 500]]
        broken += [Response(b'', headers={'Content-Length':str(LIMIT+1)}), Response(b'', chunks=[b'x'*LIMIT, b'x']),
                   Response(b'{invalid'), Response(json.dumps(dict(remark='runtime error', elements=[])).encode()),
                   Response(b'', stream_error=True)]
        current = source(proposal['queries'][0]); current['elements'][0]['timestamp'] = proposal['queries'][0]['cutoff']
        broken.append(Response(json.dumps(current).encode()))
        for i, reply in enumerate(broken):
            d = parent/f'bad{i}'; c = Session([reply], d)
            failed = run(proposal, d, approved=True, session=c, wait=lambda _: None)
            assert len(c.calls) == 1 and c.closed and reply.closed and failed['receipts'][0]['state'] == 'failed'
            assert len(failed['receipts']) == 1 and not (d/'acquisition.lock').exists()
            second = Session([], d)
            assert run(proposal, d, approved=True, session=second) == failed and not second.calls
            assert all(p.stat().st_size <= LIMIT for p in d.glob('*.json') if p.name != 'receipts.json')
        d = parent/'timeout'; c = Session([], d, fail=TimeoutError('synthetic timeout'))
        failed = run(proposal, d, approved=True, session=c)
        assert len(c.calls) == 1 and c.closed and failed['receipts'][0]['error_type'] == 'TimeoutError'
    print('E280 fake-only exact6 transport/spacing/resume/immutability/approval/scope/pre-send reservation/locks/sticky failure/cap/closure checks PASS; actual HTTP0')


if __name__ == '__main__':
    with patch('requests.adapters.HTTPAdapter.send', side_effect=RuntimeError('REAL HTTP FORBIDDEN')):
        tests()
