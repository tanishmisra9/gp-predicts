"""E271 fixed-history arithmetic, missingness, cutoffs and rejection controls."""
import copy

import numpy as np
import pandas as pd

from experiments.e271_supplier_winner import NEW, add_history, score, source_rows


def rejects(fn):
    try:
        fn()
    except (AssertionError, ValueError):
        return
    raise AssertionError('invalid source accepted')


def fixture():
    rows = []
    for k, (year, rnd, date) in enumerate([(2019, 1, '2019-01-06'), (2019, 2, '2019-01-13'),
                                         (2019, 3, '2019-01-20'), (2020, 1, '2020-01-05')]):
        for team, points in [('mercedes', 10.), ('williams', 20.), ('ferrari', 0.)]:
            for i in range(2):
                rows.append(dict(season=year, round=rnd, k=k, date=pd.Timestamp(date),
                                 code=team+str(i), driver=team+str(i), team=team, points=points))
    results = pd.DataFrame(rows)
    X = results.drop(columns='points')
    source = results[['season', 'round', 'code']].copy()
    source['brand'] = ['mercedes' if t != 'ferrari' else 'ferrari' for t in results.team]
    return X, results, source


def main():
    X, results, source = fixture()
    actual, teams = add_history(X, results, source)
    w = .5**.1
    value = actual[actual.k.eq(1) & actual.team.eq('mercedes')][NEW[0]]
    np.testing.assert_allclose(value, w*.4/(5+w), atol=1e-15, rtol=0)
    assert actual.loc[actual.k.isin([0, 3]), NEW].eq(0).all().all()
    assert actual.loc[actual.k.isin([0, 3]), 'supplier_brand'].isna().all()
    assert actual[NEW[0]].equals(actual[NEW[1]])
    assert len(teams) == 4*3*2
    # Same source race is a single peer-team observation even with two cars.
    np.testing.assert_allclose(teams[(teams.kind == 'supplier') & teams.k.eq(0) & teams.team.eq('williams')].z.iloc[0], .4, atol=1e-15)
    changed = source.copy(); changed.loc[changed.season.eq(2019) & changed['round'].ge(2), 'brand'] = 'futureΩ'
    after, _ = add_history(X, results, changed)
    pd.testing.assert_frame_equal(actual.loc[X.k.le(1), NEW], after.loc[X.k.le(1), NEW])
    prefix, _ = add_history(X, results, source[source['round'].eq(1)])
    pd.testing.assert_frame_equal(actual.loc[X.k.le(1), NEW], prefix.loc[X.k.le(1), NEW])
    changed = source.copy(); changed.loc[results.k.eq(0) & results.team.eq('williams'), 'brand'] = 'distinct'
    earlier, _ = add_history(X, results, changed)
    assert earlier.loc[X.k.eq(1), NEW[0]].ne(actual.loc[X.k.eq(1), NEW[0]]).any()
    changed_results = results.copy(); changed_results.loc[results.k.eq(0) & results.team.eq('williams'), 'points'] += 25
    earlier, _ = add_history(X, changed_results, source)
    assert earlier.loc[X.k.eq(1), NEW[0]].ne(actual.loc[X.k.eq(1), NEW[0]]).any()
    empty, _ = add_history(X, results, source.iloc[:0])
    assert empty[NEW[0]].eq(0).all() and empty.supplier_brand.isna().all()
    single_brand = source.copy(); single_brand['brand'] = results.team
    no_peers, _ = add_history(X, results, single_brand)
    assert no_peers[NEW[0]].eq(0).all()
    partial = source.drop(index=2)
    partial_result, partial_teams = add_history(X, results, partial)
    assert partial_result.loc[X.k.eq(1), 'supplier_brand'].notna().all()
    np.testing.assert_allclose(partial_teams[(partial_teams.kind == 'supplier') & partial_teams.k.eq(0) & partial_teams.team.eq('williams')].z.iloc[0], .4, atol=1e-15)
    unknown = source.copy(); unknown.loc[results.team.eq('mercedes'), 'brand'] = None
    no_membership, _ = add_history(X, results, unknown)
    assert no_membership.loc[X.team.eq('mercedes'), NEW[0]].eq(0).all()
    late_results = results.copy(); late_X = X.copy()
    # A nominal earlier event on target Thursday is not strictly before the cutoff.
    late_results.loc[results.k.eq(0), 'date'] = pd.Timestamp('2019-01-10')
    late_X.loc[X.k.eq(0), 'date'] = pd.Timestamp('2019-01-10')
    dated, _ = add_history(late_X, late_results, source)
    assert dated.loc[X.k.eq(1), NEW].eq(0).all().all()
    for bad in [pd.concat([source, source.iloc[:1]]), source.assign(season=2023), source.assign(code='unknown')]:
        rejects(lambda: add_history(X, results, bad))
    conflicting = source.copy(); conflicting.loc[0, 'brand'] = 'conflict'
    rejects(lambda: add_history(X, results, conflicting))
    rejects(lambda: add_history(X, results.assign(points=np.inf), source))
    rejects(lambda: add_history(X, results.assign(points=-1), source))
    rejects(lambda: add_history(X, results.assign(date=pd.NaT), source))
    row = dict(driverId='driver', driverNumber='1', constructorId='a', engineManufacturerId='brand', engineId=None)
    event = dict(season=2019, round=1, rows=[row])
    identity = dict(season=2019, round=1, identities={'1': 'mercedes0'})
    projected = source_rows(results, [event], [identity])
    assert projected.brand.tolist() == ['brand']
    missing = dict(identity, identities={})
    assert source_rows(results, [event], [missing]).empty
    rejects(lambda: source_rows(results, [event, event], [identity]))
    rejects(lambda: source_rows(results, [event], [dict(identity, identities={'1': 'unknown'})]))
    duplicate = copy.deepcopy(event); duplicate['rows'].append(row)
    rejects(lambda: source_rows(results, [duplicate], [identity]))
    rejects(lambda: source_rows(results, [event], [dict(identity, identities={'1': 'mercedes0', '2': 'mercedes0'})]))
    train = X[X.k.eq(0)].assign(win=[1, 0, 0, 0, 0, 0], history=np.arange(6))
    target = X[X.k.eq(1)].assign(win=0, history=np.arange(6))
    for bad in [train.iloc[:0], pd.concat([train, target]), pd.concat([train, train.iloc[:1]])]:
        rejects(lambda: score(bad, target, ['history']))
    rejects(lambda: score(train, target, ['gridn']))
    a = score(train, target, ['history'])
    b = score(train, target.assign(win=999, order=-99, gridn=-99), ['history'])
    np.testing.assert_array_equal(a, b)
    print('E271 synthetic peer arithmetic/own-team exclusion/missing/newseason/prefix/cutoff/partial/duplicate/conflict/guard PASS')


if __name__ == '__main__':
    main()
