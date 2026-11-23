"""E273 millisecond/reset/phase/dedup support boundary checks before data."""
import json

from experiments.e273_best_withdrawals import PILOT, analyse, gate, milliseconds, project


def body(patches):
    return ''.join(f'00:00:{i:02d}.000'+json.dumps({'Lines': {'1': p}})+'\r\n' for i, p in enumerate(patches)).encode()


def best(value, phase=None):
    row = {'BestLapTime': {'Value': value}}
    if phase is not None:
        row['BestLapTimes'] = {str(phase): {'arbitrary': 'excluded'}}
    return row


def fails(fn):
    try:
        fn()
    except (AssertionError, ValueError):
        return
    raise AssertionError('malformed source accepted')


def main():
    from tests.test_best_withdrawals_artifact import indexed_analysis
    roster = {'1': 'AAA'}
    patches = [best('1:30.000', 0), best('1:20.000'), best('1:30.000'),
               best('1:20.000'), best('1:30.000'), best('1:30.000')]
    projected = project(body(patches), roster); result = analyse(projected, roster)
    assert indexed_analysis(projected, roster) == result
    assert result['valid_best_updates'] == 6 and result['upward_events'] == 2
    assert result['unique_restorations'] == 1 and result['events'][1]['duplicate']
    assert result['events'][0]['old_ms'] == 80000 and result['events'][0]['new_ms'] == 90000
    for ending in [best('1:40.000'), best('1:30.000', 1), best('1:20.000'), best('1:10.000')]:
        assert analyse(project(body(patches[:2]+[ending]), roster), roster)['unique_restorations'] == 0
    for reset in [best(''), {'BestLapTime': {}}, {'BestLapTimes': {}}, {'BestLapTimes': {'0': {}, '1': {}}},
                  {'BestLapTimes': [{'Value': '1:00.000'}, {'Value': '2:00.000'}, {}]}]:
        changed = patches[:2]+[reset, best('1:30.000')]
        assert analyse(project(body(changed), roster), roster)['unique_restorations'] == 0
    unknown = [best('1:30.000'), best('1:20.000'), best('1:30.000')]
    assert analyse(project(body(unknown), roster), roster)['unique_restorations'] == 0
    two_phases = patches[:3]+[best('1:30.000', 1), best('1:20.000'), best('1:30.000')]
    assert analyse(project(body(two_phases), roster), roster)['unique_restorations'] == 2
    for raw in [body(two_phases), body([{'BestLapTimes': [{}, {}, {}]}]+patches), body(unknown)]:
        projection = project(raw, roster)
        assert indexed_analysis(projection, roster) == analyse(projection, roster)
    poison = [dict(p, Message={'poison': [999]}, Position=-999, Grid=999) for p in patches]
    assert project(body(poison), roster) == projected
    late = patches+[best('1:11.111')]
    assert analyse(project(body(late), roster), roster)['events'][:2] == result['events']
    assert project(b'', roster) == {'records': 0, 'updates': []}
    assert analyse(project(b'', roster), roster)['unique_restorations'] == 0
    assert milliseconds('1:00.001') == 60001 and milliseconds('00:01:00.001', True) == 60001
    for bad in ['0:00.000', '1:60.000', '1:20.00', True, -1]:
        fails(lambda: milliseconds(bad))
    fails(lambda: project(body(patches), {'2': 'BBB'}))
    fails(lambda: project(body(patches), {'1': 'AAA', '2': 'AAA'}))
    fails(lambda: project(b'00:00:01.000[]\r\n', roster))
    fails(lambda: project(body([{'BestLapTimes': {'3': {}}}]), roster))
    fails(lambda: project(body([{'BestLapTimes': True}]), roster))
    fails(lambda: project(body([{'BestLapTime': True}]), roster))
    fails(lambda: project(b'00:00:02.000{}\r\n00:00:01.000{}\r\n', roster))
    tied = body(patches[:3]).replace(b'00:00:01.000', b'00:00:00.000')
    assert analyse(project(tied, roster), roster)['unique_restorations'] == 1
    for cars, count, expected in [(10, 100, True), (10, 99, False), (9, 100, False)]:
        wire = []
        for i in range(count):
            car = str(i % cars+1)
            patch = best('1:20.000' if cars <= i < 2*cars else '1:30.000', 0 if i < cars else None)
            wire.append(f'00:{i//60:02d}:{i%60:02d}.000'+json.dumps({'Lines': {car: patch}})+'\r\n')
        people = {str(i+1): f'D{i}' for i in range(cars)}
        assert analyse(project(''.join(wire).encode(), people), people)['event_gate'] == expected
    reports = [dict(season=y, round=r, available=True, event_gate=True,
                    unique_restorations=2, restoration_codes=[f'D{i}', f'D{i+1}']) for i, (y, r) in enumerate(PILOT)]
    assert gate(reports)
    for key, value in [('available', False), ('unique_restorations', 0), ('restoration_codes', [])]:
        changed = [dict(r) for r in reports]; changed[0][key] = value
        assert not gate(changed)
    reports[0]['event_gate'] = reports[1]['event_gate'] = False
    assert not gate(reports)
    print('E273 synthetic clocks/known restoration/phase-blank reset/unknown/nonrestored/dedup/poison/tied-order/count gates PASS')


if __name__ == '__main__':
    main()
