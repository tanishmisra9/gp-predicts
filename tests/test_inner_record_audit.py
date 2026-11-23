"""Synthetic fail-capable checks for E284; no actual source audit at import."""
import csv
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import experiments.e284_inner_record_audit as A
from experiments.e284_inner_record_audit import calendar, metrics, paired, picks, scores, verify_row


def rejects(call):
    try:
        call()
    except (AssertionError, ValueError, KeyError):
        return
    raise AssertionError('malformed case accepted')


def fixture():
    return [dict(season='2020', round=str(rnd), date=f'2020-01-{day:02}',
                 driver=driver, code=driver.upper(), team='t'+str(i//2),
                 order=str(i+1), grid=str(3-i))
            for rnd, day in ((1, 5), (2, 12), (3, 19))
            for i, driver in enumerate(('a', 'b', 'c'))]


def test_scalar_and_edges():
    rows = fixture(); events = calendar(rows)
    source = {((2020, 1), 'A'): 0., ((2020, 2), 'A'): 2.,
              ((2020, 3), 'A'): -999., ((2020, 1), 'B'): math.nan}
    assert all(math.isinf(x) for x in scores(events, source, (2020, 1)).values())
    result = scores(events, source, (2020, 3))
    expected = 2*.5**.1/(.5**.2+.5**.1)
    assert math.isclose(result['a'], expected, abs_tol=1e-14)
    assert math.isinf(result['b']) and math.isinf(result['c'])
    assert picks(result) == ['a', 'b', 'c']
    assert picks({'é': 0., 'a': 0., 'b': math.inf}) == ['a', 'é', 'b']
    assert scores(events, {}, (2020, 2)) == dict.fromkeys(('a', 'b', 'c'), math.inf)
    changed = dict(source); changed[((2020, 1), 'A')] = 10.
    assert scores(events, changed, (2020, 3))['a'] > result['a']
    rejects(lambda: scores(events, {((2020, 9), 'A'): 0.}, (2020, 3)))
    rejects(lambda: scores(events, {((2020, 1), 'A'): math.inf}, (2020, 3)))
    bad = fixture(); bad[3]['date'] = bad[4]['date'] = bad[5]['date'] = '2020-01-08'
    rejects(lambda: scores(calendar(bad), source, (2020, 2)))
    rejects(lambda: calendar([]))
    rejects(lambda: calendar(rows+[rows[0]]))
    rejects(lambda: calendar(rows[1:]))
    bad = fixture(); bad[3]['driver'] = 'different-driver-with-reused-code'
    rejects(lambda: calendar(bad))
    for field, value in (('code', 'B'), ('driver', 'b'), ('order', '2'),
                         ('date', '2021-01-05'), ('date', 'not-a-date')):
        bad = fixture(); bad[0] = dict(bad[0], **{field: value})
        rejects(lambda: calendar(bad))


def test_source_boundary():
    with TemporaryDirectory() as directory:
        root = Path(directory); (root/'data/parts').mkdir(parents=True)
        def write(filename, records):
            columns = ['season', 'round', 'code']+list(dict(A.FILES)[filename])
            with (root/'data/parts'/filename).open('w') as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader(); writer.writerows(records)
        base = dict(season='2020', round='1', code='A')
        for i, (filename, mapping) in enumerate(A.FILES):
            write(filename, [] if i == 1 else [dict(base, **dict.fromkeys(mapping, '0'))])
        events = calendar(fixture())
        with patch.object(A, 'ROOT', root):
            assert all(list(v.values()) == [0.] for v in A.sources(events).values())
            filename, mapping = A.FILES[0]
            row = dict(base, **dict.fromkeys(mapping, '0'))
            write(filename, [row, row])
            rejects(lambda: A.sources(events))
            for change in (dict(code='unknown'), dict(round='9'),
                           dict(med_gap='inf'), dict(med_gap='malformed')):
                write(filename, [dict(row, **change)])
                rejects(lambda: A.sources(events))
            write(filename, [row])
            write(A.FILES[1][0], [row])
            rejects(lambda: A.sources(events))


def test_saved_selection_and_metrics():
    event = calendar(fixture())[(2020, 1)]
    assert metrics(event, ['a', 'b', 'c']) == dict(winner=1, podium=1., team_hit=1)
    assert metrics(event, ['b', 'a', 'c']) == dict(winner=0, podium=1., team_hit=1)
    saved = dict(pred1='a', pred2='b', pred3='c', winner='1', podium='1', team_hit='1')
    assert verify_row(event, dict(a=0., b=1., c=2.), saved)['winner'] == 1
    # The saved-selection and metric checks must fail under actual tampering.
    rejects(lambda: verify_row(event, dict(a=0., b=1., c=2.), dict(saved, pred1='b')))
    rejects(lambda: verify_row(event, dict(a=0., b=1., c=2.), dict(saved, winner='0')))
    rejects(lambda: metrics(event, ['a', 'a', 'c']))
    rejects(lambda: metrics(event, ['a', 'b', 'unknown']))
    rejects(lambda: metrics(event, ['a', 'b']))


def test_bootstrap():
    assert paired([0]*61, [0]*61) == dict(difference=0., ci95=[0., 0.], p_nonpositive=1.)
    assert paired([1]*61, [0]*61) == dict(difference=1., ci95=[1., 1.], p_nonpositive=0.)
    rejects(lambda: paired([1], [0]))


if __name__ == '__main__':
    test_scalar_and_edges()
    test_saved_selection_and_metrics()
    test_source_boundary()
    test_bootstrap()
    print('E284 synthetic weighted/missing/tie/cutoff/cohort/tamper/bootstrap checks PASS')
