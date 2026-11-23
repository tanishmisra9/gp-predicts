"""Offline E280 historical node/way contracts and exact six-query approval scope."""
import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from experiments.e280_osm_proposal import manifest, project
from f1.data import ROOT


def main():
    old = '2019-01-01T00:00:00Z'; as_of = '2019-05-08T23:59:59Z'; cutoff = '2019-05-09T00:00:00Z'
    raw = dict(osm3s=dict(timestamp_osm_base='2026-11-23T00:00:00Z'), elements=[
        dict(type='node', id=1, version=2, timestamp=old, lat=41.5, lon=2.2),
        dict(type='node', id=2, version=1, timestamp=as_of, lat=41.6, lon=2.3),
        dict(type='way', id=1, version=1, timestamp=old, nodes=[1, 2, 1], tags=dict(highway='raceway', name='ignore'))])
    expected = project(raw, as_of, cutoff)
    ignored = copy.deepcopy(raw)
    for e in ignored['elements']:
        e.update(user='poison', uid=-99, changeset=-99, comment='poison', url='poison')
        e.setdefault('tags', {}).update(name='poison', note='poison', description='poison')
    assert project(ignored, as_of, cutoff) == expected
    changed = copy.deepcopy(raw); changed['elements'][0]['lat'] += .01
    assert project(changed, as_of, cutoff) != expected  # Same way version can have different geometry.
    assert changed['elements'][-1] == raw['elements'][-1]
    bads = []
    for field, value in [('timestamp', cutoff), ('timestamp', '2019-05-08'), ('lat', float('nan')), ('lon', 181), ('version', 0), ('id', True)]:
        bad = copy.deepcopy(raw); bad['elements'][0][field] = value; bads.append(bad)
    bad = copy.deepcopy(raw); bad['elements'].append(bad['elements'][0]); bads.append(bad)
    bad = copy.deepcopy(raw); bad['elements'][-1]['nodes'] = [1, 3]; bads.append(bad)
    bad = copy.deepcopy(raw); bad['elements'][-1]['nodes'] = [1, 1]; bads.append(bad)
    bad = copy.deepcopy(raw); bad['elements'][-1]['tags']['highway'] = 'service'; bads.append(bad)
    bad = copy.deepcopy(raw); bad['elements'][0]['type'] = 'relation'; bads.append(bad)
    bad = copy.deepcopy(raw); bad['elements'].append(dict(type='node', id=3, version=1, timestamp=old, lat=0., lon=0.)); bads.append(bad)
    bads += [dict(elements=[]), dict(raw, remark='runtime error')]
    for bad in bads:
        try: project(bad, as_of, cutoff)
        except (AssertionError, ValueError): pass
        else: raise AssertionError('malformed/history-unsafe snapshot accepted')
    with patch('requests.adapters.HTTPAdapter.send', side_effect=RuntimeError('all HTTP forbidden')):
        proposal = manifest()
    assert not proposal['approval_received'] and not proposal['acquisition_performed']
    exact = [('catalunya', 2019, 5, '2019-05-09'), ('catalunya', 2022, 6, '2022-05-19'),
             ('hungaroring', 2019, 12, '2019-08-01'), ('hungaroring', 2022, 13, '2022-07-28'),
             ('monza', 2019, 14, '2019-09-05'), ('monza', 2022, 16, '2022-09-08')]
    coords = dict(catalunya=(41.57, 2.26111), hungaroring=(47.5789, 19.2486), monza=(45.6156, 9.28111))
    assert len(proposal['queries']) == 6
    for q, (circuit, year, rnd, date) in zip(proposal['queries'], exact):
        assert (q['circuit'], q['season'], q['round']) == (circuit, year, rnd)
        assert q['cutoff'] == date+'T00:00:00Z'
        boundary = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
        assert q['as_of'] == (boundary-timedelta(seconds=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
        lat, lon = coords[circuit]; box = [round(lat-.035, 6), round(lon-.035, 6), round(lat+.035, 6), round(lon+.035, 6)]
        assert q['bbox'] == box and q['endpoint'] == 'https://overpass-api.de/api/interpreter'
        query = '[out:json][timeout:25][maxsize:67108864][date:"'+q['as_of']+'"];way["highway"="raceway"]('+','.join(format(n, '.6f') for n in box)+');(._;>;);out meta;'
        assert q['query'] == query and q['query_sha256'] == hashlib.sha256(query.encode()).hexdigest()
    path = ROOT / 'results/e280_osm_query_proposal.json'
    if path.exists(): assert json.loads(path.read_text()) == proposal
    print('E280 all6 independent dates/bboxes/queries/approval-pending scope; historical node/way/text poison/positive node response/14 negatives PASS; HTTP0')


if __name__ == '__main__':
    main()
