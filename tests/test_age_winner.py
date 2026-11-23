"""E275 age, prior-DOB/count boundaries and train-only scorer guards."""
import copy
import datetime as dt

import numpy as np
import pandas as pd

from experiments.e275_age_winner import AGES, EXPERIENCE, NEW, add_history, cutoff_date, score


def rejects(fn):
    try:
        fn()
    except (AssertionError, ValueError, KeyError):
        return
    raise AssertionError('invalid inputs accepted')


def fixture():
    events = [(2019, i+1, pd.Timestamp('2019-01-06')+pd.Timedelta(days=7*i)) for i in range(12)]
    events += [(2020, 1, pd.Timestamp('2020-01-05'))]
    X = pd.DataFrame([dict(season=y, round=r, date=date, k=k, driver=driver, win=int(driver == 'a'))
                      for k, (y, r, date) in enumerate(events) for driver in ['a', 'b']])
    source = [dict(season=r.season, round=r.round, date=r.date.strftime('%Y-%m-%d'), driver=r.driver,
                   dateOfBirth='1980-02-29' if r.driver == 'a' else '1995-01-01') for r in X.itertuples()]
    return X, source


def main():
    X, source = fixture(); actual = add_history(X, source)
    assert actual.loc[X.k.eq(0), AGES].isna().all().all() and actual.loc[X.k.eq(0), EXPERIENCE[0]].eq(0).all()
    assert actual.loc[X.k.eq(9), EXPERIENCE[0]].eq(9).all() and actual.loc[X.k.eq(9), EXPERIENCE[1]].eq(1).all()
    assert actual.loc[X.k.eq(10), EXPERIENCE[0]].eq(10).all() and actual.loc[X.k.eq(10), EXPERIENCE[1]].eq(0).all()
    assert actual.loc[X.k.eq(12), EXPERIENCE[0]].eq(12).all()
    wanted = (dt.date(2019, 1, 10)-dt.date(1980, 2, 29)).days/365.2425
    row = actual[X.k.eq(1) & X.driver.eq('a')].iloc[0]
    assert row.prior_age_years == wanted and row.prior_age_squared == wanted**2
    assert cutoff_date(pd.Timestamp('2020-03-01')) == pd.Timestamp('2020-02-27')
    assert cutoff_date(pd.Timestamp('2020-01-05')) == pd.Timestamp('2020-01-02')
    changed = copy.deepcopy(source)
    for r in changed:
        if r['round'] >= 10 or r['season'] == 2020:
            r['dateOfBirth'] = '1999-01-01'
    after = add_history(X, changed)
    pd.testing.assert_frame_equal(actual.loc[X.k.le(9), NEW], after.loc[X.k.le(9), NEW])
    prefix = add_history(X[X.k.le(9)].reset_index(drop=True), [r for r in source if r['season'] == 2019 and r['round'] < 10])
    pd.testing.assert_frame_equal(actual.loc[X.k.le(9), NEW], prefix[NEW])
    missing = add_history(X, [])
    assert missing[AGES].isna().all().all() and missing[EXPERIENCE].equals(actual[EXPERIENCE])
    empty = [dict(r, dateOfBirth='') for r in source]
    assert add_history(X, empty)[AGES].isna().all().all() and add_history(X, empty).prior_dob.isna().all()
    conflicting = copy.deepcopy(source); conflicting[0]['dateOfBirth'] = '1980-03-01'
    unknown = add_history(X, conflicting)
    assert unknown.loc[X.driver.eq('a') & X.k.eq(12), AGES].isna().all().all()
    early = [dict(r, dateOfBirth='1980-03-01') if r['driver'] == 'a' and r['season'] == 2019 else r for r in source]
    assert not add_history(X, early).loc[X.k.eq(12), AGES].equals(actual.loc[X.k.eq(12), AGES])
    less = X.drop(index=0).reset_index(drop=True)
    less_source = [r for r in source if not (r['season'] == 2019 and r['round'] == 1 and r['driver'] == 'a')]
    less = add_history(less, less_source)
    assert less[less.k.eq(12) & less.driver.eq('a')].observed_entry_count.iloc[0] == 11
    boundary_X = X.copy(); boundary_source = copy.deepcopy(source)
    boundary_X.loc[X.k.eq(0), 'date'] = pd.Timestamp('2019-01-10')
    for r in boundary_source:
        if r['season'] == 2019 and r['round'] == 1:
            r['date'] = '2019-01-10'
    boundary = add_history(boundary_X, boundary_source)
    assert boundary.loc[X.k.eq(1), EXPERIENCE[0]].eq(0).all() and boundary.loc[X.k.eq(1), AGES].isna().all().all()
    for bad in [source+[source[0]], [dict(source[0], season=2023)], [dict(source[0], driver='unknown')],
                [dict(source[0], date='2019-01-07')], [dict(source[0], dateOfBirth='not-a-date')]]:
        rejects(lambda: add_history(X, bad))
    rejects(lambda: add_history(pd.concat([X, X.iloc[:1]], ignore_index=True), source))
    rejects(lambda: add_history(X.assign(date=pd.NaT), source))
    train = actual[actual.k.lt(12)]; current = actual[actual.k.eq(12)]
    for bad in [train.iloc[:0], pd.concat([train, current]), pd.concat([train, train.iloc[:1]])]:
        rejects(lambda: score(bad, current, NEW))
    rejects(lambda: score(train, current, NEW+['gridn']))
    np.testing.assert_array_equal(score(train, current, NEW), score(train, current.assign(win=999, order=-1, gridn=-999), NEW))
    print('E275 synthetic age/square/10entry/leap/Thursday/unknown/conflict/prefix/earlier/count/scorer guards PASS')


if __name__ == '__main__':
    main()
