"""E181 numerical and real-source cutoff checks, before winner scoring."""
import numpy as np
import pandas as pd

from experiments.e181_joint_filter import filter_predictions, likelihood, observation


def approx_derivative(fun, x):
    step = np.eye(len(x)) * 1e-5
    return np.array([(fun(x + e) - fun(x - e)) / 2e-5 for e in step]).T


def test_derivatives():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(5, 7)); B = rng.normal(size=(3, 7)); y = rng.normal(size=3)
    ranking = np.array([3, 1, 4, 0, 2]); x = rng.normal(size=7)
    value, grad, hess = likelihood(x, A, B, y, ranking)
    numeric_grad = approx_derivative(lambda z: likelihood(z, A, B, y, ranking)[0], x).ravel()
    numeric_hess = approx_derivative(lambda z: likelihood(z, A, B, y, ranking)[1], x)
    np.testing.assert_allclose(grad, numeric_grad, atol=1e-7)
    np.testing.assert_allclose(hess, numeric_hess, atol=1e-7)
    assert np.linalg.eigvalsh(hess).min() > -1e-10
    assert np.isfinite(value)


def test_edges():
    g = pd.DataFrame(dict(season=[2019]*3, round=[1]*3, driver=['a','b','c'], team=['x','x','y'],
                          order=[1,2,3], q1=[80.,80.,np.nan], q2=[np.nan]*3, q3=[np.nan]*3))
    assert filter_predictions(g.iloc[:0]).empty
    B, y, _ = observation(g, np.eye(3))
    assert B.shape == (0,3) and len(y) == 0
    g.q1 = np.nan
    g.team = 'x'
    p = filter_predictions(g)
    np.testing.assert_allclose(p.prior_joint_win, 1/3, atol=.02)
    assert p.prior_joint_win.sum() == 1 or np.isclose(p.prior_joint_win.sum(), 1)
    one = g.iloc[:1].copy()
    assert filter_predictions(one).prior_joint_win.iloc[0] == 1
    try:
        filter_predictions(pd.concat([g,g]))
    except AssertionError:
        pass
    else:
        raise AssertionError('duplicate identity accepted')


def test_source_cutoff():
    df = pd.read_csv('data/results.csv').loc[lambda d: d.season.eq(2019) & d['round'].le(5)].copy()
    before = filter_predictions(df)
    rng = np.random.default_rng(91)
    target = 4
    changed = df.copy()
    for _, ids in changed[changed['round'].ge(target)].groupby('round').groups.items():
        changed.loc[ids, ['q1','q2','q3']] = rng.uniform(30,300,(len(ids),3))
        changed.loc[ids, 'order'] = rng.permutation(np.arange(1,len(ids)+1))
        changed.loc[ids, 'status'] = 'Engine'
        changed.loc[ids, 'laps'] = 0
    after = filter_predictions(changed)
    np.testing.assert_array_equal(before[before['round'].le(target)].prior_joint_win,
                                  after[after['round'].le(target)].prior_joint_win)
    assert not np.allclose(before[before['round'].eq(5)].prior_joint_win,
                           after[after['round'].eq(5)].prior_joint_win)
    np.testing.assert_allclose(before.groupby(['season','round']).prior_joint_win.sum(), 1)
    # Source row order cannot import finishing-position ordering into forecast ties.
    reordered = filter_predictions(df.sample(frac=1, random_state=3))
    np.testing.assert_array_equal(before.prior_joint_win, reordered.prior_joint_win)
    for column in ('q1', 'order'):
        isolated = df.copy()
        mask = isolated['round'].ge(target)
        if column == 'order':
            for _, ids in isolated[mask].groupby('round').groups.items():
                isolated.loc[ids, 'order'] = isolated.loc[ids, 'order'].to_numpy()[::-1]
        else:
            isolated.loc[mask, ['q1', 'q2', 'q3']] = rng.uniform(30,300,(mask.sum(),3))
        p = filter_predictions(isolated)
        np.testing.assert_array_equal(before[before['round'].le(target)].prior_joint_win,
                                      p[p['round'].le(target)].prior_joint_win)
        assert not np.allclose(before[before['round'].eq(5)].prior_joint_win,
                               p[p['round'].eq(5)].prior_joint_win)
    null = filter_predictions(df, shuffled=True)
    np.testing.assert_array_equal(before[before['round'].eq(1)].prior_joint_win,
                                  null[null['round'].eq(1)].prior_joint_win)


if __name__ == '__main__':
    test_derivatives()
    test_edges()
    test_source_cutoff()
    print('joint likelihood derivatives, missing observations, source cutoff and responsiveness passed')
