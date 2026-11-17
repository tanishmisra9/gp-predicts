"""Reconstruct the E190 availability gate from saved disjoint pair records."""
import json
from pathlib import Path

import pandas as pd


def test_saved_pairs():
    result=json.loads(Path('results/e190_cache_feasibility.json').read_text())
    pairs=pd.read_csv('results/e190_traffic_pairs.csv')
    sessions=result['sessions']
    assert [(r['season'],r['round']) for r in sessions]==[(2019,5),(2019,12),(2019,14)]
    assert len(pairs)==344 and pairs.season.eq(2019).all()
    for report in sessions:
        p=pairs[pairs.season.eq(report['season'])&pairs['round'].eq(report['round'])]
        assert report['available'] and not report['missing_fields']
        assert len(p)==report['pairs']
        counts=p.groupby('driver').size().reindex(report['by_driver'],fill_value=0)
        assert counts.to_dict()==report['by_driver']
        assert int(counts.ge(2).sum())==report['drivers_with_two_pairs']
        assert p.entering.sum()==report['entering'] and (~p.entering).sum()==report['leaving']
        for _,g in p.groupby('driver'):
            laps=list(g.lap1)+list(g.lap2)
            assert len(laps)==len(set(laps))
            assert g.lap2.sub(g.lap1).eq(1).all()
    gate=all(r['available'] for r in sessions) and len(pairs)>=60
    gate=gate and sum(r['pairs']>=20 for r in sessions)>=2
    gate=gate and sum(r['drivers_with_two_pairs']>=r['drivers']/2 for r in sessions)>=2
    assert gate==result['gate'] and gate


if __name__=='__main__':
    test_saved_pairs()
    print('all E190 session/driver/direction counts, pair disjointness and feasibility gate reproduce')
