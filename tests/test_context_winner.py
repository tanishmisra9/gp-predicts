"""E277 independent calculus, exchangeability and raw-source temporal checks."""
import math

import numpy as np
import pandas as pd

from experiments.e277_context_winner import fit_score, forward, initial, inputs, labels, load_inputs, make_features, objective


def scalar_forward(theta, A, context):
    p = A.shape[1]; offset = 8*p
    U = np.array(theta[:offset]).reshape(p, 8); a = theta[offset:offset+8]; offset += 8
    V = np.array(theta[offset:offset+128]).reshape(16, 8); offset += 128
    b = theta[offset:offset+8]; v = theta[offset+8:]
    H = [[math.tanh(sum(float(x)*float(w) for x, w in zip(row, U[:, j]))+a[j]) for j in range(8)] for row in A]
    output = []
    for i, h in enumerate(H):
        c = [sum(other[j] for k, other in enumerate(H) if k != i)/(len(H)-1) for j in range(8)] if context and len(H)>1 else [0.]*8
        z = h+c
        output.append(sum(math.tanh(sum(float(x)*float(w) for x, w in zip(z, V[:, j]))+b[j])*v[j] for j in range(8)))
    return np.array(output)


def synthetic():
    rng = np.random.default_rng(91); A = rng.normal(size=(7, 3)); groups = np.array([0, 0, 0, 1, 1, 1, 1])
    y = np.array([0., 1, 0, 0, 0, 1, 0]); weights = np.array([.4, .6]); theta = initial(3)
    for context in [True, False]:
        loss, gradient = objective(theta, A, groups, y, weights, context)
        numeric = []
        for j in range(len(theta)):
            delta = np.zeros_like(theta); delta[j] = 1e-5
            numeric.append((objective(theta+delta, A, groups, y, weights, context)[0]-objective(theta-delta, A, groups, y, weights, context)[0])/2e-5)
        np.testing.assert_allclose(gradient, numeric, atol=2e-10, rtol=2e-7)
        manual_loss = .005*sum(float(t)**2 for t in theta)
        for k, indices in enumerate([np.arange(3), np.arange(3, 7)]):
            direct = scalar_forward(theta, A[indices], context)
            np.testing.assert_allclose(forward(theta, A[indices], np.zeros(len(indices), int), context)[0], direct, atol=1e-15)
            manual_loss += weights[k]*(math.log(sum(math.exp(float(u)) for u in direct))-float(direct[np.argmax(y[indices])]))
        np.testing.assert_allclose(loss, manual_loss, atol=1e-15)
        zero_loss, _ = objective(theta*0, A, groups, y, weights, context)
        np.testing.assert_allclose(zero_loss, .4*math.log(3)+.6*math.log(4), atol=1e-15)
    singleton, _ = forward(theta, A[:1], np.zeros(1, int))
    np.testing.assert_allclose(singleton, scalar_forward(theta, A[:1], True), atol=1e-15)
    changed = A[:3].copy(); changed[-1] += 5
    before, _ = forward(theta, A[:3], np.zeros(3, int))
    after, _ = forward(theta, changed, np.zeros(3, int))
    assert before[0] != after[0]
    before, _ = forward(theta, A[:3], np.zeros(3, int), False)
    after, _ = forward(theta, changed, np.zeros(3, int), False)
    assert before[0] == after[0]
    rows = [dict(k=k, driver=f'd{i}', date=pd.Timestamp('2019-01-06')+pd.Timedelta(days=7*k), win=int(i==0), history=float(i), missing=np.nan)
            for k in range(5) for i in range(3)]
    tr = pd.DataFrame(rows); te = tr.iloc[-3:].copy().assign(k=5, date=pd.Timestamp('2019-02-10'))
    F = ['history', 'missing']; prior, a, b = inputs(tr, te, F)
    assert a.shape == (15, 2) and (a[:, 1] == 0).all() and (b[:, 1] == 0).all()
    np.testing.assert_array_equal(labels(prior, True)[:9], labels(prior.iloc[:9], True))
    for bad in [tr.iloc[:0], pd.concat([tr, te]), pd.concat([tr, tr.iloc[:1]]), tr.assign(date=te.date.iloc[0])]:
        try: inputs(bad, te, F)
        except AssertionError: pass
        else: raise AssertionError('invalid training accepted')
    try: inputs(tr, te.assign(gridn=1), F+['gridn'])
    except ValueError: pass
    else: raise AssertionError('forbidden feature accepted')
    try: fit_score(tr.assign(win=0), te, F)
    except AssertionError: pass
    else: raise AssertionError('missing winner accepted')
    for context, shuffled in [(True, False), (False, False), (True, True)]:
        score, _ = fit_score(tr, te, F, context, shuffled)
        again, _ = fit_score(tr.sample(frac=1, random_state=2), te.assign(win=999, gridn=-999), F, context, shuffled)
        np.testing.assert_array_equal(score, again)
        reordered, _ = fit_score(tr, te.iloc[::-1], F, context, shuffled)
        np.testing.assert_allclose(score, reordered[::-1], atol=1e-15, rtol=0)
    print('E277 independent scalar forward/loss, ALL-parameter central gradients, singleton/missing/guards/null/permutation/context response PASS', flush=True)


def real_source():
    results, pace = load_inputs(); X, F = make_features(results, pace)
    target = X.season.eq(2022) & X['round'].eq(10); k = X.loc[target, 'k'].iloc[0]
    changed = results.copy(); future = changed.season.eq(2022) & changed['round'].ge(10)
    for col in ['grid', 'q_pos', 's_grid', 's_order', 'order', 'finish', 'points', 'laps', 'q1', 'q2', 'q3']:
        changed.loc[future, col] = 999
    changed.loc[future, 'status'] = 'poison'
    later = changed.season.eq(2022) & changed['round'].gt(10)
    for col in ['driver', 'team', 'circuit']:
        changed.loc[later, col] = changed.loc[later, col].astype(str)+'futureΩ'
    cp = pace.copy(); cp.loc[cp.season.eq(2022) & cp['round'].ge(10), 'pace_gap'] = 999.
    Z, ZF = make_features(changed, cp); assert F == ZF
    pd.testing.assert_frame_equal(X.loc[X.k.le(k), F], Z.loc[Z.k.le(k), F])
    pr = results[results.season.lt(2022) | (results.season.eq(2022) & results['round'].le(10))]
    pp = pace[pace.season.lt(2022) | (pace.season.eq(2022) & pace['round'].le(10))]
    prefix, PF = make_features(pr, pp); assert F == PF
    pd.testing.assert_frame_equal(X.loc[X.k.le(k), F], prefix[F])
    train = X.k.lt(k) & X.season.ge(2019)
    baseline = None
    for context, shuffled in [(True, False), (False, False), (True, True)]:
        a, _ = fit_score(X[train], X[target], F, context, shuffled)
        b, _ = fit_score(Z[train], Z[target], F, context, shuffled)
        c, _ = fit_score(prefix[prefix.k.lt(k) & prefix.season.ge(2019)], prefix[prefix.k.eq(k)], F, context, shuffled)
        np.testing.assert_array_equal(a, b); np.testing.assert_array_equal(a, c)
        if context and not shuffled: baseline = a
    changed = results.copy(); prior = changed.season.eq(2022) & changed['round'].eq(9)
    changed.loc[prior, 'order'] = int(prior.sum())+1-changed.loc[prior, 'order']; changed.loc[prior, 'points'] += 100
    Z, _ = make_features(changed, pace)
    responsive, _ = fit_score(Z[train], Z[target], F)
    assert not np.array_equal(responsive, baseline)
    print('E277 actual raw current/future outcomes/grid/quali/sprint/pace/identities + prefix EXACT features/3 fitted models; earlier fitted responsiveness PASS', flush=True)


if __name__ == '__main__':
    synthetic(); real_source()
