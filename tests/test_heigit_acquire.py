"""Fake-only E285 transport tests; envelopes are NOT real Parquet files."""
import copy
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from experiments.e285_heigit_acquire import run
from experiments.e285_heigit_proposal import ENDPOINT, proposal
from tests.test_heigit_proposal import line
from tests.test_osm_acquire import Response, Session

DUMMY = 'SYNTHETIC-AUTH-DO-NOT-PERSIST'


def body(query):
    west, south, east, north = query['aoi']
    x, y = (west+east)/2, (south+north)/2
    row = dict(osm_type='way', osm_id=1, version=1, minor_version=0,
               edit_timestamp=query['as_of'], clipped=False, tags={'highway': 'raceway'},
               geom_type='LineString', geom=line([[x, y], [x+.001, y+.001]]).hex())
    # Test fixture only: deliberately not a Parquet encoding/decoder.
    return b'PAR1'+json.dumps([row]).encode()+b'PAR1'


def decode(raw):
    rows = json.loads(raw[4:-4])
    for row in rows:
        row['geom'] = bytes.fromhex(row['geom'])
    return rows, 'OGC:CRS84'


def rejects(call, error=AssertionError):
    try:
        call()
    except error:
        return
    raise AssertionError('expected rejection')


def test_runner():
    plan = proposal(); limit = plan['limits']['max_body_bytes']
    def replies():
        return [Response(body(q)) for q in plan['queries']]
    def acquire(folder, client, **kw):
        return run(plan, folder, approved=True, key_provider=lambda: DUMMY,
                   decode=decode, session=client, wait=lambda _: None, **kw)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        key = Mock(return_value=DUMMY)
        folder = root/'no-approval'; client = Session([], folder)
        rejects(lambda: run(plan, folder, key_provider=key, decode=decode, session=client))
        changed = copy.deepcopy(plan); changed['queries'][0]['request']['endpoint'] += '/changed'
        rejects(lambda: run(changed, folder, approved=True, key_provider=key, decode=decode, session=client))
        assert not folder.exists() and not client.calls and not key.called
        folder = root/'success'; responses = replies(); client = Session(responses, folder); gaps = []
        key = Mock(return_value=DUMMY)
        state = run(plan, folder, approved=True, key_provider=key, decode=decode, session=client, wait=gaps.append)
        assert key.call_count == 1 and len(client.calls) == 6 and gaps == [5]*5
        assert client.closed and all(r.closed for r in responses)
        assert len(state['receipts']) == 6 and all(r['state'] == 'success' for r in state['receipts'])
        for (url, kw), q in zip(client.calls, plan['queries']):
            assert url == ENDPOINT
            assert kw == dict(params=q['request']['params'], stream=True, allow_redirects=False,
                              timeout=(10, 35), headers={'Accept': 'application/octet-stream',
                                                        'Authorization': DUMMY,
                                                        'User-Agent': 'gp-predicts-research/E285'})
        assert (folder/'SOURCE_NOTICE.txt').exists()
        for path in folder.iterdir():
            assert DUMMY.encode() not in path.read_bytes()
        before = {p.name: p.read_bytes() for p in folder.iterdir()}
        no_send = Session([], folder); no_key = Mock(side_effect=AssertionError('unexpected key read'))
        assert run(plan, folder, approved=True, key_provider=no_key, decode=decode, session=no_send) == state
        assert not no_key.called and not no_send.calls
        assert {p.name: p.read_bytes() for p in folder.iterdir()} == before
        # Clean prefixes resume only keys never attempted; all previous files immutable.
        partial = copy.deepcopy(state); partial['receipts'] = partial['receipts'][:2]
        for receipt in state['receipts'][2:]:
            for name in ('raw_file', 'projection_file'):
                (folder/receipt[name]).unlink()
        (folder/'receipts.json').write_text(json.dumps(partial))
        resumed = Session(replies()[2:], folder); gaps = []
        done = run(plan, folder, approved=True, key_provider=lambda: DUMMY, decode=decode,
                   session=resumed, wait=gaps.append)
        assert len(resumed.calls) == 4 and len(done['receipts']) == 6 and gaps == [5]*4
        for receipt in done['receipts'][:2]:
            for name in ('raw_file', 'projection_file'):
                assert (folder/receipt[name]).read_bytes() == before[receipt[name]]
        # Uncertain interrupted checkpoints are sticky and do not need a key.
        interrupted = copy.deepcopy(done); interrupted['receipts'] = interrupted['receipts'][:3]
        interrupted['receipts'][-1] = dict(index=2, state='started', request_sha256=plan['queries'][2]['request_sha256'])
        (folder/'receipts.json').write_text(json.dumps(interrupted))
        client = Session([], folder); key = Mock()
        assert run(plan, folder, approved=True, key_provider=key, decode=decode, session=client) == interrupted
        assert not client.calls and not key.called
        (folder/'acquisition.lock').write_text('live/stale owner: no automatic removal')
        rejects(lambda: acquire(folder, client), FileExistsError)
        assert (folder/'acquisition.lock').exists() and not client.calls
        (folder/'acquisition.lock').unlink()
        # A valid prefix with an orphan body, or corrupted canonical bytes, cannot refetch.
        (folder/'receipts.json').write_text(json.dumps(partial)); key = Mock()
        rejects(lambda: run(plan, folder, approved=True, key_provider=key, decode=decode, session=client))
        assert not client.calls and not key.called
        (folder/'receipts.json').write_text(json.dumps(done))
        raw_path = folder/done['receipts'][0]['raw_file']; saved = raw_path.read_bytes()
        raw_path.write_bytes(saved+b'X')
        rejects(lambda: acquire(folder, client), RuntimeError)
        assert not client.calls; raw_path.write_bytes(saved)
        bad_path = copy.deepcopy(done); bad_path['receipts'][0]['raw_file'] = '../unrelated.env'
        (folder/'receipts.json').write_text(json.dumps(bad_path))
        rejects(lambda: acquire(folder, client))
        assert not client.calls
        # Inclusive accepted-file boundary uses a synthetic envelope, not Parquet roundtrip.
        folder = root/'boundary'; responses = replies(); original = responses[0].payload
        responses[0].payload = original[:-4]+b' '*(limit-len(original))+original[-4:]
        client = Session(responses, folder)
        assert acquire(folder, client)['receipts'][0]['bytes'] == limit
        assert len(client.calls) == 6
        # Every transport/content/projection failure globally stops and is sticky.
        bad = [Response(b'', status=status) for status in (301, 401, 403, 429, 500)]
        bad += [Response(b'', headers={'Content-Length': str(limit+1)}),
                Response(b'', headers={'Content-Length': '-1'}),
                Response(b'', chunks=[b'x'*limit, b'x']), Response(b'', stream_error=True),
                Response(b'not-parquet'), Response(b'PAR1invalid-jsonPAR1')]
        future = json.loads(body(plan['queries'][0])[4:-4]); future[0]['edit_timestamp'] = plan['queries'][0]['cutoff']
        bad.append(Response(b'PAR1'+json.dumps(future).encode()+b'PAR1'))
        for i, response in enumerate(bad):
            folder = root/f'bad{i}'; client = Session([response], folder)
            failed = acquire(folder, client)
            assert len(client.calls) == 1 and client.closed and response.closed
            assert len(failed['receipts']) == 1 and failed['receipts'][0]['state'] == 'failed'
            client = Session([], folder); key = Mock()
            assert run(plan, folder, approved=True, key_provider=key, decode=decode, session=client) == failed
            assert not client.calls and not key.called and not (folder/'acquisition.lock').exists()
        # Key-like values in exception messages must not escape through stdout or receipts.
        folder = root/'echo'; client = Session([], folder, fail=TimeoutError('Authorization: '+DUMMY))
        stream = io.StringIO()
        with redirect_stdout(stream):
            failed = acquire(folder, client)
        assert failed['receipts'][0]['error_type'] == 'TimeoutError'
        assert DUMMY not in stream.getvalue()+json.dumps(failed)
        for path in folder.iterdir():
            assert DUMMY.encode() not in path.read_bytes()
        folder = root/'key-failure'; client = Session([], folder)
        with redirect_stdout(io.StringIO()):
            try:
                run(plan, folder, approved=True, key_provider=Mock(side_effect=ValueError(DUMMY)),
                    decode=decode, session=client)
            except RuntimeError as exc:
                assert DUMMY not in str(exc)
            else:
                raise AssertionError('credential exception accepted')
        assert not client.calls and client.closed and not (folder/'receipts.json').exists()
        # Real default-session configuration is tested with its send still forbidden.
        folder = root/'session-config'
        from requests import Session as RequestsSession
        real = RequestsSession(); real.get = Mock(side_effect=RuntimeError(DUMMY))
        with patch('requests.Session', return_value=real):
            run(plan, folder, approved=True, key_provider=lambda: DUMMY, decode=decode)
        assert real.trust_env is False and real.get_adapter(ENDPOINT).max_retries.total == 0
        # An interruption retains its pre-send checkpoint rather than pretending terminal failure.
        folder = root/'interrupted'; client = Session([], folder, fail=KeyboardInterrupt())
        rejects(lambda: acquire(folder, client), KeyboardInterrupt)
        assert json.loads((folder/'receipts.json').read_text())['receipts'][0]['state'] == 'started'
        assert client.closed and not (folder/'acquisition.lock').exists()
        no_key = Mock(); client = Session([], folder)
        run(plan, folder, approved=True, key_provider=no_key, decode=decode, session=client)
        assert not no_key.called and not client.calls


if __name__ == '__main__':
    with patch('requests.adapters.HTTPAdapter.send', side_effect=AssertionError('REAL HTTP FORBIDDEN')) as send:
        test_runner()
        send.assert_not_called()
    print('E285 fake-only transport/checkpoint/resume/stop/cap/dummy-key redaction PASS; real HTTP and Parquet decodes ZERO')
