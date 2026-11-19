"""Sparse/list index preservation, fixed lap-axis mapping and label-free source guards."""
import json
from experiments.e231_driver_score_structure import skeleton,indexed,decode,structure


class Poison(dict):
    def __getitem__(self,key):
        if key=='Message':raise AssertionError('text accessed')
        return super().__getitem__(key)


def test_structure():
    assert skeleton(Poison({'Message':object(),'1':[[1,'2.5','not numeric',False]]}))=={'1':[[1.,2.5,None,None]]}
    assert indexed([None,[1.,2.]])==[(0,None),(1,[1.,2.])]
    assert indexed({'3':[4.]})==[(3,[4.])]
    assert indexed(None)==[]
    roster={str(n) for n in range(1,11)}
    scores={car:[[lap,50+lap,80-lap] for lap in range(1,11)] for car in roster}
    raw=('00:30:00.000'+json.dumps({'Scores':scores})+'\r\n').encode()
    records,obs,meta=decode(raw,roster)
    assert records[0]['scores']['1'][3][2]==76. and len(obs)==300 and not meta['unknown_car_keys']
    patch=('00:31:00.000'+json.dumps({'Scores':{'1':{'3':{'1':'66'}}}})+'\r\n').encode()
    records,obs,meta=decode(raw+patch,roster)
    assert records[1]['scores']['1']=={'3':{'1':66.}}
    laps=[dict(car=car,lap=lap,time=lap*60.) for car in roster for lap in range(1,11)]
    stats=structure(obs,laps)
    assert stats['structure_gate'] and stats['axis_match_fraction']==1 and stats['nonaxis_components']==[1,2]
    assert stats['revised_coordinate_count']==1 and stats['drivers_with_10_rows_2_components']==10
    changed=[o|{'value':99.} if o['component']==0 else o for o in obs]
    assert not structure(changed,laps)['structure_gate']
    assert not structure([],[])['structure_gate']
    assert not decode(raw,{'1'})[2]['unknown_car_keys']==[]
    try:indexed({'not_index':1})
    except ValueError:pass
    else:raise AssertionError('nonindexed schema accepted')


if __name__=='__main__':test_structure();print('DriverScore index/sparse-patch/lap-axis/empty/text tests pass')
