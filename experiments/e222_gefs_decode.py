"""Six-message decoder pilot. Run in the isolated requirements-weather environment."""
import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import requests

from experiments.e220_gefs_metadata import before_cutoff
from experiments.e221_gefs_headers import identification

ROOT = Path(__file__).resolve().parents[1]
LAT, LON = 41.57, 2.26111  # existing data/circuit_static.csv, Catalunya
KEYS = ('edition', 'discipline', 'parameterCategory', 'parameterNumber', 'shortName', 'units',
        'typeOfLevel', 'level', 'dataDate', 'dataTime', 'validityDate', 'validityTime',
        'startStep', 'endStep', 'stepUnits', 'stepType', 'gridType', 'Ni', 'Nj',
        'iDirectionIncrementInDegrees', 'jDirectionIncrementInDegrees', 'packingType',
        'numberOfMissing', 'numberOfDataPoints')


def validate(metadata, point, variable, initialization):
    init = datetime.fromisoformat(initialization)
    common = (metadata['edition'] == 2 and metadata['discipline'] == 0
              and metadata['dataDate'] == int(init.strftime('%Y%m%d')) and metadata['dataTime'] == 1200
              and metadata['validityDate'] == int((init + timedelta(hours=96)).strftime('%Y%m%d'))
              and metadata['validityTime'] == 1200 and str(metadata['stepUnits']) == '1'
              and metadata['endStep'] == 96 and metadata['gridType'] == 'regular_ll'
              and math.isfinite(point['value']) and 0 <= point['distance'] <= 80)
    if variable == 'TMP':
        field = (metadata['parameterCategory'] == 0 and metadata['parameterNumber'] == 0
                 and metadata['units'] == 'K' and metadata['typeOfLevel'] == 'heightAboveGround'
                 and metadata['level'] == 2 and metadata['startStep'] == 96
                 and metadata['stepType'] == 'instant' and 180 <= point['value'] <= 330)
    elif variable == 'APCP':
        field = (metadata['parameterCategory'] == 1 and metadata['parameterNumber'] == 8
                 and metadata['units'] == 'kg m**-2' and metadata['typeOfLevel'] == 'surface'
                 and metadata['startStep'] == 90 and metadata['stepType'] == 'accum'
                 and 0 <= point['value'] <= 500)
    else:
        raise ValueError('unregistered variable')
    return bool(common and field)


def decode(data):
    import eccodes as E
    handle = E.codes_new_from_message(data)
    try:
        metadata = {key: E.codes_get(handle, key) for key in KEYS}
        point = dict(E.codes_grib_find_nearest(handle, LAT, LON, npoints=1)[0])
        return metadata, point
    finally:
        E.codes_release(handle)


def main():
    import eccodes as E
    prior = json.loads((ROOT / 'results/e220_gefs_metadata.json').read_text())
    receipts = {r['url']: r for r in prior['rows'] if r['modification_before_cutoff']}
    source = json.loads((ROOT / 'results/e221_gefs_headers.json').read_text())
    if not source['source_gate'] or len(source['rows']) != 3:
        raise ValueError('requires passing E221 pilot')
    rows = []
    for source_row in source['rows']:
        receipt = receipts[source_row['url']]
        inv = source_row['inventory']
        for number, variable in ((63, 'TMP'), (69, 'APCP')):
            item = inv[number - 1]
            if item['number'] != number or item['variable'] != variable:
                raise ValueError('inventory no longer matches registration')
            start, end = item['offset'], inv[number]['offset'] - 1
            size = end - start + 1
            if not 37 <= size <= 1024 * 1024:
                raise ValueError('message exceeds registered size bound')
            row = dict(url=source_row['url'], variable=variable, start=start, end=end,
                       initialization=receipt['initialization'], cutoff=receipt['cutoff'])
            try:
                with requests.get(row['url'], headers={'Range': f'bytes={start}-{end}',
                                  'If-Match': receipt['etag'], 'Accept-Encoding': 'identity'},
                                  timeout=20, stream=True, allow_redirects=False) as response:
                    row['status'] = response.status_code
                    row['last_modified'] = response.headers.get('Last-Modified')
                    if (response.status_code != 206 or response.headers.get('ETag') != receipt['etag']
                            or response.headers.get('Content-Range') != f"bytes {start}-{end}/{receipt['content_length']}"
                            or not before_cutoff(dict(status=200, content_length=receipt['content_length'],
                                                     cutoff=receipt['cutoff'], last_modified=row['last_modified']))):
                        raise ValueError('unconfirmed range/object/cutoff')
                    data = response.raw.read(size + 1)
                if len(data) != size or data[-4:] != b'7777':
                    raise ValueError('truncated or overlong message')
                ident = identification(data[:37])
                if (ident['first_message_length'] != size or ident['reference'] != row['initialization']
                        or ident['significance'] != 1 or ident['production'] != 0 or ident['data_type'] != 3):
                    raise ValueError('message identity mismatch')
                tag = row['initialization'][:10].replace('-', '')
                path = ROOT / f'results/e222_{tag}_{variable}.grib2'
                path.write_bytes(data)
                row.update(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(data).hexdigest(), bytes=size)
                metadata, point = decode(data)
                row.update(metadata=metadata, nearest=point,
                           gate=validate(metadata, point, variable, row['initialization']))
                row['converted_value'] = point['value'] - 273.15 if variable == 'TMP' else point['value']
                row['converted_unit'] = 'C' if variable == 'TMP' else 'mm water equivalent / 6h'
            except (requests.RequestException, ValueError, OSError, E.CodesInternalError) as exc:
                row.update(error=type(exc).__name__ + ': ' + str(exc), gate=False)
            rows.append(row)
    result = dict(rows=rows, source_gate=len(rows) == 6 and all(r['gate'] for r in rows),
                  eccodes_python_version=E.__version__, eccodes_c_version=E.codes_get_api_version(),
                  winner_utility_tested=False, race_window_alignment_established=False)
    (ROOT / 'results/e222_gefs_decode.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
