"""E181: online Laplace driver/team filter; completed-event observations only."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

import f1.walkforward as W
from f1.data import DATA, load_results
from f1.features import ERA, build_features, pre_weekend_cols
from experiments.e168_winner import win_score


def likelihood(x, A, B, y, ranking):
    residual = B @ x - y
    value = residual @ residual / 2
    grad = B.T @ residual
    hess = B.T @ B
    for j in range(len(ranking) - 1):
        C = A[ranking[j:]]
        u = C @ x
        p = softmax(u)
        mean = p @ C
        value += logsumexp(u) - u[0]
        grad += mean - C[0]
        hess += C.T @ (p[:, None] * C) - np.outer(mean, mean)
    return value, grad, hess


def observation(g, A, rng=None):
    q = g[['q1', 'q2', 'q3']].min(axis=1).to_numpy()
    ranking = np.argsort(g.order.to_numpy(), kind='stable')
    if rng is not None:
        permutation = rng.permutation(len(g))
        q = q[permutation]
        ranking = np.argsort(g.order.to_numpy()[permutation], kind='stable')
    valid = np.isfinite(q)
    if valid.sum() < 2 or np.std(q[valid]) == 0:
        return A[:0], np.empty(0), ranking
    B = A[valid] - A[valid].mean(axis=0)
    y = -(q[valid] - q[valid].mean()) / q[valid].std()
    return B, y, ranking


def filter_predictions(source, shuffled=False):
    required = ['season', 'round', 'driver', 'team', 'order', 'q1', 'q2', 'q3']
    assert not source.duplicated(['season', 'round', 'driver']).any()
    assert source[required[:5]].notna().all().all()
    assert source.season.between(2019, 2022).all(), 'E181 is inner-only'
    identities, m, V, rows = {}, np.empty(0), np.empty((0, 0)), []
    rng = np.random.default_rng(181)
    previous_era = None
    for (season, round_), g in source.groupby(['season', 'round'], sort=True):
        g = g.sort_values('driver').reset_index(drop=True)
        assert sorted(g.order) == list(range(1, len(g) + 1))
        for kind in ('driver', 'team'):
            for name in sorted(g[kind].unique()):
                key = (kind, name)
                if key not in identities:
                    identities[key] = len(m)
                    m = np.append(m, 0.)
                    V = np.pad(V, ((0, 1), (0, 1)))
                    V[-1, -1] = .25 if kind == 'driver' else 1.
        team = np.array([kind == 'team' for kind, _ in identities])
        if previous_era is not None:
            V += np.diag(np.where(team, .04, .0025))
            if ERA(season) != previous_era:
                scale = np.where(team, .5, 1.)
                m *= scale
                V *= np.outer(scale, scale)
                V += np.diag(team.astype(float))
        previous_era = ERA(season)
        A = np.zeros((len(g), len(m)))
        for i, row in g.iterrows():
            for kind in ('driver', 'team'):
                A[i, identities[(kind, row[kind])]] = 1.
        # Independent prediction RNG: observation null cannot alter the forecast draws.
        pred_rng = np.random.default_rng(181000 + 100 * int(season) + int(round_))
        samples = m + pred_rng.standard_normal((2048, len(m))) @ np.linalg.cholesky(V).T
        probability = softmax(samples @ A.T, axis=1).mean(axis=0)
        assert np.isfinite(probability).all() and np.isclose(probability.sum(), 1)
        rows.append(g[['season', 'round', 'driver']].assign(prior_joint_win=probability))
        B, y, ranking = observation(g, A, rng if shuffled else None)
        precision = np.linalg.inv(V)

        def objective(x):
            value, grad, _ = likelihood(x, A, B, y, ranking)
            d = x - m
            return value + d @ precision @ d / 2, grad + precision @ d

        fitted = minimize(objective, m, jac=True, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-11, 'gtol': 1e-7})
        assert fitted.success, fitted.message
        _, _, hess = likelihood(fitted.x, A, B, y, ranking)
        m = fitted.x
        V = np.linalg.inv(precision + hess)
        V = (V + V.T) / 2
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['season', 'round', 'driver', 'prior_joint_win'])


def main():
    df = load_results().loc[lambda d: d.season.le(2022)]
    pace = pd.read_csv(DATA / 'racepace_ff1.csv').loc[lambda d: d.season.le(2022)]
    X = build_features(df, pace).loc[lambda d: d.season.ge(2019)]; base = pre_weekend_cols(X)
    source = df[df.season.ge(2019)]
    for shuffled, column in ((False, 'prior_joint_win'), (True, 'prior_joint_null')):
        prediction = filter_predictions(source, shuffled).rename(columns={'prior_joint_win': column})
        X = X.merge(prediction, on=['season', 'round', 'driver'], validate='one_to_one')
    F = base + ['prior_joint_win', 'prior_joint_null']; W.assert_pre_weekend(F)
    W.MODELS.update(
        standings=lambda tr, te, F: (-te.d_season_pts.fillna(0)*100 + te.d10_order.fillna(11)*1e-3).to_numpy(),
        pre_logit=lambda tr, te, F: W.m_logit_pod(tr, te, base, C=.03, hl=100),
        win_logit=lambda tr, te, F: win_score(tr, te, base),
        joint_win=lambda tr, te, F: -te.prior_joint_win.to_numpy(),
        joint_null=lambda tr, te, F: -te.prior_joint_null.to_numpy(),
    )
    R, P = W.walk_forward(X, [2020, 2021, 2022], ['standings', 'pre_logit', 'win_logit', 'joint_win', 'joint_null'],
                          F=F, ens={'ens': ['standings', 'pre_logit']}, tie=None)
    joined = P.merge(df[['season', 'round', 'driver', 'team', 'order']], on=['season', 'round', 'driver'], validate='many_to_one')
    winner = df[df.order.eq(1)][['season', 'round', 'team']].rename(columns={'team': 'winner_team'})
    picked = joined.sort_values(['season', 'round', 'model', 'score', 'driver']).drop_duplicates(['season', 'round', 'model'])
    picked = picked.merge(winner, on=['season', 'round'], validate='many_to_one')
    team_hits = picked.assign(hit=picked.team.eq(picked.winner_team)).groupby('model').hit.sum()
    print(R.groupby('model')[['winner', 'podium']].agg(['sum', 'mean']).round(4).to_string(), flush=True)
    print('winning team', team_hits.to_dict(), flush=True)
    print(R.pivot_table(index='season', columns='model', values='winner', aggfunc='sum').to_string(), flush=True)
    a = R[R.model.eq('joint_win')].set_index(['season', 'round']).winner
    for ref in ('standings', 'ens', 'win_logit'):
        b = R[R.model.eq(ref)].set_index(['season', 'round']).winner
        print('joint_win minus', ref, W.paired_boot(a, b), flush=True)
    probs = joined[joined.model.isin(['joint_win', 'joint_null', 'win_logit', 'pre_logit'])].copy()
    probs['prob'] = -probs.score / (-probs.score).groupby([probs.season, probs['round'], probs.model]).transform('sum')
    loss = probs[probs.order.eq(1)].assign(loss=lambda d: -np.log(d.prob.clip(lower=1e-15))).groupby('model').loss.mean()
    print('winner log-loss', loss.to_dict(), flush=True)
    hits = R.groupby('model').winner.sum()
    gate = hits.joint_win >= 35 and team_hits.joint_win >= team_hits.win_logit and loss.joint_win < loss.win_logit and hits.joint_win > hits.joint_null
    print('INNER GATE', bool(gate), flush=True)
    assert R.season.max() == 2022 and R.groupby('model').size().eq(61).all()
    R.to_csv('results/e181_inner_per_race.csv', index=False)
    P.to_csv('results/e181_inner_predictions.csv', index=False)


if __name__ == '__main__':
    main()
