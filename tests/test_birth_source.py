"""E274 synthetic source projection, date validation and causal DOB lookups."""
import copy
import json

from experiments.e274_birth_source import birth_date, page_metadata, plan, prior_records, project, summarize


def fails(fn):
    try:
        fn()
    except (AssertionError, ValueError, KeyError):
        return
    raise AssertionError('invalid metadata accepted')


def main():
    official = [dict(season=y, round=1, date=f'{y}-01-07', driver=d)
                for y in range(2018, 2023) for d in ['a', 'b'] if y > 2018 or d == 'a']
    rows = [dict(r, dateOfBirth='1980-02-29' if r['driver'] == 'a' else '1990-07-01') for r in official]
    summary, prior = summarize(rows, official)
    assert summary['content_gate'] and summary['unique_drivers'] == summary['unique_covered'] == 2
    assert all(r['prior_dob'] is not None for r in prior)
    target = [r for r in official if r['season'] == 2020]
    original = prior_records(rows, target)
    future = [dict(r, dateOfBirth='1998-05-05') if r['season'] >= 2020 else r for r in rows]
    assert prior_records(future, target) == original
    assert prior_records([r for r in rows if r['season'] < 2020], target) == original
    earlier = [dict(r, dateOfBirth='1980-03-01') if r['season'] < 2020 and r['driver'] == 'a' else r for r in rows]
    assert prior_records(earlier, target) != original
    assert prior_records(rows, [official[0]])[0]['prior_dob'] is None
    changed = copy.deepcopy(rows); changed[0]['dateOfBirth'] = '1980-03-01'
    conflicts, conflicting_prior = summarize(changed, official)
    assert not conflicts['content_gate'] and set(conflicts['conflicts']) == {'a'}
    assert any(r['prior_conflict'] for r in conflicting_prior)
    missing, _ = summarize(rows[:-1], official)
    assert not missing['content_gate'] and len(missing['missing_entries']) == 1
    fails(lambda: summarize(rows+[rows[0]], official))
    fails(lambda: summarize(rows+[dict(rows[0], driver='unknown')], official))
    assert birth_date(None, '2018-01-07') is None and birth_date('', '2018-01-07') is None
    assert birth_date('1980-02-29', '2018-01-07') == '1980-02-29'
    for invalid in ['1981-02-29', '1980-2-29', '2020-01-01', '2010-01-01', '1900-01-01', True, 1980]:
        fails(lambda: birth_date(invalid, '2018-01-07'))
    item = plan()[0]; assert len(plan()) == 24
    assert [(y, sum(r['season'] == y for r in plan())) for y in range(2018, 2023)] == [(2018, 5), (2019, 5), (2020, 4), (2021, 5), (2022, 5)]
    race = dict(season='2018', round='1', date='2018-01-07', Results=[dict(Driver=dict(driverId='a', dateOfBirth='1980-02-29'), position='1', grid='1', points='25', status='text')])
    payload = dict(MRData=dict(limit='100', offset='0', total='420', RaceTable=dict(season='2018', Races=[race])))
    races = page_metadata(json.dumps(payload), item)
    mapping = {(r['season'], r['round'], r['driver']): r['date'] for r in official}
    projected = project(races, mapping)
    poison = copy.deepcopy(races)
    for field in ['position', 'grid', 'points', 'status']:
        poison[0]['Results'][0][field] = {'poison': [999]}
    poison[0]['Results'][0]['Driver']['givenName'] = {'poison': [999]}
    assert project(poison, mapping) == projected
    changed = copy.deepcopy(races); changed[0]['Results'][0]['Driver']['dateOfBirth'] = '1980-03-01'
    assert project(changed, mapping) != projected
    for field, value in [('limit', '99'), ('offset', '100'), ('total', '419')]:
        wrong = copy.deepcopy(payload); wrong['MRData'][field] = value
        fails(lambda: page_metadata(json.dumps(wrong), item))
    wrong = copy.deepcopy(payload); wrong['MRData']['RaceTable']['Races'][0]['season'] = '2026'
    fails(lambda: page_metadata(json.dumps(wrong), item))
    fails(lambda: project(races+races, mapping))
    wrong = copy.deepcopy(races); wrong[0]['Results'][0]['Driver']['driverId'] = 'unknown'
    fails(lambda: project(wrong, mapping))
    wrong = copy.deepcopy(races); wrong[0]['date'] = '2018-01-08'
    fails(lambda: project(wrong, mapping))
    print('E274 synthetic birth/leap/missing/bounds/pagination/scope/duplicates/conflict/raw-poison/prior-prefix/earlier-response PASS')


if __name__ == '__main__':
    main()
