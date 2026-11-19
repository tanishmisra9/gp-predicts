"""E239 transactional prior positions, fixed actor 2 and same-patch reversals."""
from experiments.e239_overtake_marker import analyse


def record(data):
    return dict(seconds=0.,numeric=[dict(path=[car,key],value=float(value)) for car,fields in data.items() for key,value in fields.items()])


def test_marker():
    roster={'1','2'}
    initial=record({'1':{'Position':2},'2':{'Position':1}})
    passed=record({'1':{'Position':1,'OvertakeState':2},'2':{'Position':2,'OvertakeState':1}})
    reset=record({'1':{'Position':2,'OvertakeState':0},'2':{'Position':1,'OvertakeState':0}})
    summary,events=analyse([initial]+[r for _ in range(50) for r in (passed,reset)],roster)
    assert summary['qualified'] and summary['echo_markers']==50 and summary['actor_gains']==50
    assert events[0]['before']==2 and events[0]['after']==1 and events[0]['counterparts']==['2']
    summary,events=analyse([passed],roster)
    assert not summary['qualified'] and summary['known_prior_markers']==0 and not events[0]['echo']
    for changed in [record({'1':{'Position':2,'OvertakeState':2},'2':{'Position':1}}),
        record({'1':{'Position':1,'OvertakeState':2}}),record({'1':{'OvertakeState':2}}),
        record({'1':{'Position':1,'OvertakeState':1},'2':{'Position':2}})]:
        summary,_=analyse([initial,changed],roster)
        assert not summary['qualified'] and summary['echo_markers']==0
    assert analyse([],roster)[0]['state2_markers']==0
    for invalid in [record({'3':{'Position':1}}),record({'1':{'OvertakeState':3}}),record({'1':{'Position':1.5}})]:
        try: analyse([invalid],roster)
        except ValueError: pass
        else: raise AssertionError('invalid schema accepted')
    print('E239 fixed-state/prior-before-patch/reversal/no-counterpart/empty/schema checks pass')


if __name__=='__main__':
    test_marker()
