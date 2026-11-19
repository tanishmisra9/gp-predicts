"""Recorded index/status/name/gate consistency; raw index intentionally not retained."""
import json
from f1.data import ROOT


def test_artifact():
    data=json.loads((ROOT/'results/e229_pit_feed_index.json').read_text())
    assert {(r['season'],r['round']) for r in data['sessions']}=={(2019,5),(2019,12),(2019,14)}
    good=0
    for r in data['sessions']:
        assert r['available'] and r['status']==200
        assert r['url'].startswith('https://livetiming.formula1.com/static/2019/') and r['url'].endswith('_Race/Index.json')
        assert r['requests']==[dict(url=r['url'],status=200)]
        names=r['feeds'];assert names==sorted(set(names)) and len(names)<=500
        expected=sorted(set(names)&{'PitStopSeries','PitStopTime','PitStopData'})
        assert expected==r['pit_candidates']==[]
        assert 'PitLaneTimeCollection' in names
        good+=bool(expected)
        print(r['season'],r['round'],'feed names',len(names),'pit candidates',len(expected))
    assert data['gate']==(good>=2)==False
    assert not data['pit_payload_acquired'] and not data['winner_fit']
    print('Independent recorded-source index/gate/scope audit passes; raw-source body not retained')


if __name__=='__main__':test_artifact()
