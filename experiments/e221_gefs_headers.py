"""Fixed three-object GRIB identification and structured index pilot; no field values."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests

from experiments.e220_gefs_metadata import before_cutoff

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 1024 * 1024


def identification(data):
    if len(data) != 37 or data[:4] != b'GRIB' or data[7] != 2:
        raise ValueError('not a complete GRIB2 identification header')
    length = int.from_bytes(data[8:16], 'big')
    section_length = int.from_bytes(data[16:20], 'big')
    if section_length < 21 or data[20] != 1 or length < 16 + section_length + 4:
        raise ValueError('invalid section or message length')
    stamp = datetime(int.from_bytes(data[28:30], 'big'), *data[30:35], tzinfo=timezone.utc)
    return dict(reference=stamp.isoformat(), significance=data[27], production=data[35],
                data_type=data[36], first_message_length=length)


def inventory(text):
    rows = []
    for line in text.splitlines():
        parts = line.split(':')
        if len(parts) < 7 or not parts[0].isdigit() or not parts[1].isdigit():
            raise ValueError('malformed index row')
        if not re.fullmatch(r'd=\d{10}', parts[2]):
            raise ValueError('missing index reference')
        rows.append(dict(number=int(parts[0]), offset=int(parts[1]), initialization=parts[2][2:],
                         variable=parts[3], level=parts[4], period=parts[5]))
    if not rows or [r['number'] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError('empty or nonsequential index')
    offsets = [r['offset'] for r in rows]
    if offsets[0] != 0 or any(a >= b for a, b in zip(offsets, offsets[1:])):
        raise ValueError('invalid index offsets')
    return rows


def main():
    source = json.loads((ROOT / 'results/e220_gefs_metadata.json').read_text())
    selected = [r for r in source['rows'] if r['modification_before_cutoff']]
    if len(selected) != 3 or not all(before_cutoff(r) for r in selected):
        raise ValueError('pilot requires the three registered E220 receipts')
    results = []
    for prior in selected:
        row = dict(url=prior['url'], expected_reference=prior['initialization'], cutoff=prior['cutoff'])
        try:
            with requests.get(prior['url'], headers={'Range': 'bytes=0-36', 'If-Match': prior['etag'],
                              'Accept-Encoding': 'identity'}, stream=True, timeout=20, allow_redirects=False) as response:
                row['header_status'] = response.status_code
                if (response.status_code != 206 or response.headers.get('Content-Range') !=
                        f"bytes 0-36/{prior['content_length']}" or response.headers.get('ETag') != prior['etag']):
                    raise ValueError('range or object identity not confirmed')
                data = response.raw.read(38)
                row['header_hex'] = data.hex()
                row['identification'] = identification(data)
                current = dict(status=200, content_length=prior['content_length'], cutoff=prior['cutoff'],
                               last_modified=response.headers.get('Last-Modified'))
                row['current_last_modified'] = current['last_modified']
                if not before_cutoff(current):
                    raise ValueError('current modification receipt not pre-cutoff')
        except (requests.RequestException, ValueError, OSError) as exc:
            row['header_error'] = type(exc).__name__ + ': ' + str(exc)
        try:
            with requests.get(prior['url'] + '.idx', stream=True, timeout=20,
                              allow_redirects=False, headers={'Accept-Encoding': 'identity'}) as response:
                row['index_status'] = response.status_code
                if response.status_code != 200:
                    raise ValueError('index unavailable')
                payload = response.raw.read(LIMIT + 1)
                if len(payload) > LIMIT:
                    raise ValueError('index exceeds bound')
                text = payload.decode('ascii')
                tag = prior['initialization'][:10].replace('-', '')
                (ROOT / f'results/e221_{tag}_inventory.txt').write_text(text)
                row['inventory'] = inventory(text)
        except (requests.RequestException, ValueError, OSError) as exc:
            row['index_error'] = type(exc).__name__ + ': ' + str(exc)
        ident = row.get('identification', {})
        expected = datetime.fromisoformat(prior['initialization']).strftime('%Y%m%d%H')
        inv = row.get('inventory', [])
        tmp = [r for r in inv if r['variable'] == 'TMP' and r['level'] == '2 m above ground']
        rain = [r for r in inv if r['variable'] == 'APCP' and r['level'] == 'surface']
        row['header_gate'] = ('header_error' not in row and ident.get('reference') == prior['initialization']
                              and ident.get('significance') == 1 and ident.get('production') == 0
                              and ident.get('data_type') in (1, 3))
        row['inventory_gate'] = (bool(tmp and rain) and all(r['initialization'] == expected for r in inv)
                                 and all(re.fullmatch(r'(?:\d+-)?96 hour (?:acc )?fcst', r['period'])
                                         for r in tmp + rain))
        results.append(row)
    output = dict(rows=results, source_gate=all(r['header_gate'] and r['inventory_gate'] for r in results),
                  weather_values_acquired=False, winner_utility_tested=False)
    (ROOT / 'results/e221_gefs_headers.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps([{k: v for k, v in r.items() if k != 'inventory'} for r in results], indent=2))


if __name__ == '__main__':
    main()
