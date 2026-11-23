"""E270 source projection excludes labels and rejects ambiguous identities."""
import copy
import hashlib

import pandas as pd
import yaml

from experiments.e270_supplier_records import identities, project, verified


def fails(fn):
    try:
        fn()
    except (AssertionError, yaml.YAMLError):
        return
    raise AssertionError('negative control accepted')


def main():
    row = dict(driverNumber=5, driverId='a', constructorId='b', engineManufacturerId='c',
               position=1, gridPosition=1, points=25, time='x', reasonRetired='text')
    raw = yaml.safe_dump([row]).encode()
    projection = project(raw)
    changed = copy.deepcopy(row)
    for field in ['position', 'gridPosition', 'points', 'time', 'reasonRetired']:
        changed[field] = {'arbitrary': ['poison']}
    assert project(yaml.safe_dump([changed])) == projection
    changed['engineManufacturerId'] = 'different'
    assert project(yaml.safe_dump([changed])) != projection
    fails(lambda: project(yaml.safe_dump([row, row])))
    fails(lambda: project('!!python/object:os.system [echo forbidden]'))
    fails(lambda: project(yaml.safe_dump([dict(row, driverNumber=True)])))
    assert project('- driverNumber: 5\n')[0]['engineManufacturerId'] is None
    assert identities(pd.DataFrame({'DriverNumber': ['5', '5', '44'], 'Driver': ['VET', 'VET', 'HAM']})) == {'5': 'VET', '44': 'HAM'}
    fails(lambda: identities(pd.DataFrame({'DriverNumber': ['5', '5'], 'Driver': ['A', 'B']})))
    fails(lambda: identities(pd.DataFrame({'DriverNumber': ['5', '6'], 'Driver': ['A', 'A']})))
    digest = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    assert verified(raw, digest) == raw
    fails(lambda: verified(raw+b' ', digest))
    print('E270 projection label/grid/text invariance, supplier responsiveness, duplicate identities/hash/unsafe-tag negatives PASS')


if __name__ == '__main__':
    main()
