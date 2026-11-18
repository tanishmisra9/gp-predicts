"""Known binary layouts and index boundary failures; no network."""
from experiments.e221_gefs_headers import identification, inventory


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


if __name__ == '__main__':
    test_headers()
    print('GEFS identification/index tests ok')
