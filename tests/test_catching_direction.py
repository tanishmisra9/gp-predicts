"""E241 transactional position context, emitted-only codes and fixed freshness."""
from experiments.e241_catching_direction import analyse


def record(time,data):
    return dict(seconds=time,numeric=[dict(path=[car,key],value=float(value)) for car,fields in data.items() for key,value in fields.items()])


def fixture():
    rows=[record(0,{'1':{'Position':2,'Interval':5},'2':{'Position':1},'3':{'Position':3}})]
    for k in range(1,121):
        rows.append(record(k,{'1':{'Interval':4 if k%2 else 5,'Catching':1 if k%2 else 2}}))
    return rows


def test_direction():
    roster={'1','2','3'}; rows=fixture(); result=analyse(rows,roster)
    assert result['qualified_a'] and not result['qualified_b'] and result['scored_updates']==120
    assert result['mapping_a_accuracy']==1 and result['contingency']['1_decrease']==60
    for age,accepted in [(0,True),(10,True),(10.001,False),(-1,False)]:
        sample=[rows[0],record(age,{'1':{'Interval':4,'Catching':1}})]
        assert analyse(sample,roster)['events'][0]['scored']==accepted
    assert not analyse([rows[0],record(1,{'1':{'Catching':1}})],roster)['events'][0]['scored']
    assert not analyse([rows[0],record(1,{'1':{'Interval':5,'Catching':1}})],roster)['events'][0]['scored']
    changed=record(1,{'1':{'Position':3,'Interval':4,'Catching':1},'3':{'Position':2}})
    assert not analyse([rows[0],changed],roster)['events'][0]['scored']
    swap=record(1,{'1':{'Interval':4,'Catching':1},'2':{'Position':3},'3':{'Position':1}})
    assert not analyse([rows[0],swap],roster)['events'][0]['scored']
    assert not analyse([],roster)['qualified_a']
    for invalid in (record(1,{'9':{'Catching':1}}),record(1,{'1':{'Catching':3}})):
        try: analyse([invalid],roster)
        except ValueError: pass
        else: raise AssertionError('invalid source accepted')
    print('E241 fixed mapping/transactional opponent/freshness/ties/missing/empty/source checks pass')


if __name__=='__main__':
    test_direction()
