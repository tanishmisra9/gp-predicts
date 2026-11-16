"""E184 uses completed earlier sprints, never a target weekend's points."""
import numpy as np
import pandas as pd

from experiments.e184_sprint_points import POINT_FEATURES, add_full_points


def test_totals():
    r = pd.DataFrame(dict(season=[2019,2019,2019,2019,2020,2020], round=[1,1,2,2,1,1],
        driver=['a','b']*3, team=['x','y']*3, points=[1.,2.,10.,0.,2.,1.], s_points=[3.,np.nan,0.,0.,0.,0.]))
    x = r[['season','round','driver','team']].assign(k=[0,0,1,1,2,2])
    got = add_full_points(x,r)
    np.testing.assert_allclose(got.loc[x.k.eq(1),POINT_FEATURES], [[1.,1.],[.5,.5]])
    assert got.loc[x.k.isin([0,2]),POINT_FEATURES].isna().all().all()
    pd.testing.assert_frame_equal(got[x.columns],x)
    try:
        add_full_points(x,pd.concat([r,r.iloc[:1]]))
    except AssertionError:
        pass
    else:
        raise AssertionError('duplicate source entrant accepted')


def test_real_source():
    r = pd.read_csv('data/results.csv').loc[lambda d:d.season.le(2022)].copy()
    x = r[['season','round','driver','team']].copy()
    x['k'] = x.groupby(['season','round'],sort=True).ngroup()
    before = add_full_points(x,r)
    target = (2021,12)
    mask = x.season.eq(target[0]) & x['round'].eq(target[1])
    k = x.loc[mask,'k'].iloc[0]
    future = x.k.ge(k)
    bad = r.copy()
    bad.loc[future,['points','s_points']] = 999.
    after = add_full_points(x,bad)
    pd.testing.assert_frame_equal(before.loc[mask,POINT_FEATURES],after.loc[mask,POINT_FEATURES],check_exact=True)
    # Alter the preceding sprint only, keeping all race points fixed.
    bad = r.copy()
    bad.loc[x.k.eq(k-1),'s_points'] = np.arange(x.k.eq(k-1).sum())*100.
    response = add_full_points(x,bad)
    assert not before.loc[mask,POINT_FEATURES].equals(response.loc[mask,POINT_FEATURES])
    # Independently sum race and sprint columns before each event.
    q = r.assign(k=x.k)
    for _, cur in x.groupby('k',sort=True):
        prior = q[q.k.lt(cur.k.iloc[0]) & q.season.eq(cur.season.iloc[0])]
        if prior.empty:
            assert before.loc[cur.index,POINT_FEATURES].isna().all().all()
            continue
        for key,col in zip(('driver','team'),POINT_FEATURES):
            # Separate summation of race and sprint columns, without using helper totals.
            totals = prior.groupby(key).points.sum()+prior.groupby(key).s_points.sum()
            expected = cur[key].map(totals)/max(totals.max(),1e-9)
            np.testing.assert_allclose(before.loc[cur.index,col],expected,equal_nan=True)


if __name__=='__main__':
    test_totals()
    test_real_source()
    print('prior sprint totals, season reset, source cutoff and independent sums passed')
