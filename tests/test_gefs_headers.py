"""Known binary layouts and index boundary failures; no network."""
from experiments.e221_gefs_headers import identification, inventory
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch
import experiments.e221_gefs_headers as E


def test_headers():
    data = bytearray(37)
    data[:4] = b'GRIB'; data[7] = 2; data[8:16] = (1000).to_bytes(8, 'big')
    data[16:20] = (21).to_bytes(4, 'big'); data[20] = 1; data[27] = 1
    data[28:30] = (2020).to_bytes(2, 'big'); data[30:35] = bytes([8, 12, 12, 0, 0]); data[36] = 3
    out = identification(data)
    assert out == dict(reference='2020-08-12T12:00:00+00:00', significance=1,
                       production=0, data_type=3, first_message_length=1000)
    bads = [data[:36], data + b'0']
    for offset, value in [(0, 0), (7, 1), (20, 2), (30, 13), (33, 60)]:
        bad = data.copy(); bad[offset] = value; bads.append(bad)
    for bad in bads:
        try: identification(bad)
        except ValueError: continue
        raise AssertionError('malformed binary header accepted')
    text = '1:0:d=2020081212:TMP:2 m above ground:96 hour fcst:\n2:100:d=2020081212:APCP:surface:90-96 hour acc fcst:\n'
    assert inventory(text)[1]['offset'] == 100
    for bad in ('', 'bad', text.replace('2:100:', '2:0:'), text.replace('2:100:', '3:100:'),
                text.replace('d=2020081212', 'd=bad')):
        try: inventory(bad)
        except ValueError: continue
        raise AssertionError('malformed index accepted')


def test_ignored_range_never_reads_body():
    source = json.loads((E.ROOT / 'results/e220_gefs_metadata.json').read_text())
    responses = []

    def get(url, **kwargs):
        assert kwargs['stream'] and not kwargs['allow_redirects'] and kwargs['timeout'] == 20
        response = MagicMock()
        response.__enter__.return_value = response
        response.status_code = 404 if url.endswith('.idx') else 200
        response.raw.read.side_effect = AssertionError('uncertified body read')
        responses.append(response)
        return response

    with TemporaryDirectory() as folder:
        root = Path(folder); (root / 'results').mkdir()
        (root / 'results/e220_gefs_metadata.json').write_text(json.dumps(source))
        with patch.object(E, 'ROOT', root), patch.object(E.requests, 'get', side_effect=get) as request, patch('builtins.print'):
            E.main()
        output = json.loads((root / 'results/e221_gefs_headers.json').read_text())
        assert request.call_count == 6 and not output['source_gate']
        assert all(not r.raw.read.called for r in responses)
        assert all(not r['header_gate'] and not r['inventory_gate'] for r in output['rows'])


if __name__ == '__main__':
    test_headers()
    test_ignored_range_never_reads_body()
    print('GEFS identification/index tests ok')
