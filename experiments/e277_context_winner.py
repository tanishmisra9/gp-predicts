"""E277: one fixed competitor-context network; existing pre-weekend inputs only."""
import hashlib
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import f1.walkforward as W
from experiments.e168_winner import win_score
from f1.data import DATA, ROOT
from f1.features import build_features, pre_weekend_cols

KEYS = ['season', 'round', 'driver']
CONTROLS = ['row_win', 'gbm_win', 'win_logit', 'standings', 'ens']
CONFIG = dict(width=8, l2=.01, half_life=100, init_seed=0, null_seed=277,
              maxiter=300, maxfun=10000, ftol=1e-10, gtol=1e-6)


def cutoff(date):
    date = pd.Timestamp(date).normalize()
    return date-pd.Timedelta(days=(date.weekday()-3) % 7)


def load_inputs():
    results = pd.read_csv(DATA / 'results.csv', parse_dates=['date']).loc[lambda d: d.season.between(2018, 2022)]
    pace = pd.read_csv(DATA / 'racepace_ff1.csv').loc[lambda d: d.season.between(2018, 2022)]
    return results, pace


def make_features(results, pace):
    assert results.season.between(2018, 2022).all() and pace.season.between(2018, 2022).all()
    assert not results.duplicated(KEYS).any()
    X = build_features(results, pace).merge(results[KEYS+['date']], on=KEYS, validate='one_to_one')
    F = pre_weekend_cols(X); W.assert_pre_weekend(F)
    return X, F


def shapes(p):
    return [(p, 8), (8,), (16, 8), (8,), (8,)]


def unpack(theta, p):
    output = []; start = 0
    for shape in shapes(p):
        size = int(np.prod(shape)); output.append(theta[start:start+size].reshape(shape)); start += size
    assert start == len(theta)
    return output


def initial(p):
    rng = np.random.default_rng(0)
    return np.concatenate([rng.normal(0, 1/np.sqrt(p), (p, 8)).ravel(), np.zeros(8),
                           rng.normal(0, .25, (16, 8)).ravel(), np.zeros(8), rng.normal(0, .1, 8)])


def forward(theta, A, group, context=True):
    U, a, V, b, v = unpack(theta, A.shape[1])
    H = np.tanh(A @ U+a)
    counts = np.bincount(group); sums = np.zeros((len(counts), 8)); np.add.at(sums, group, H)
    divisor = np.maximum(counts[group]-1, 1)[:, None]
    C = (sums[group]-H)/divisor if context else np.zeros_like(H)
    Z = np.concatenate([H, C], axis=1); J = np.tanh(Z @ V+b)
    return J @ v, (H, Z, J, divisor, counts)


def objective(theta, A, group, y, weights, context=True):
    utility, (H, Z, J, divisor, counts) = forward(theta, A, group, context)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    assert np.array_equal(group, np.repeat(np.arange(len(counts)), counts))
    maxima = np.maximum.reduceat(utility, starts)
    ex = np.exp(utility-maxima[group]); sums = np.add.reduceat(ex, starts)
    logs = utility-maxima[group]-np.log(sums[group]); probability = ex/sums[group]
    loss = -np.sum(weights[group]*y*logs)+.005*(theta @ theta)
    du = weights[group]*(probability-y)
    U, a, V, b, v = unpack(theta, A.shape[1])
    dJ = du[:, None]*v*(1-J*J)
    dZ = dJ @ V.T; dH = dZ[:, :8]
    if context:
        dC = dZ[:, 8:]/divisor
        total = np.zeros((len(counts), 8)); np.add.at(total, group, dC)
        dH = dH+total[group]-dC
    da = dH*(1-H*H)
    gradients = [A.T @ da, da.sum(axis=0), Z.T @ dJ, dJ.sum(axis=0), J.T @ du]
    gradient = np.concatenate([g.ravel() for g in gradients])+.01*theta
    return float(loss), gradient


def labels(train, shuffled):
    y = train.win.to_numpy(float).copy()
    assert train.groupby('k').win.sum().eq(1).all() and np.isin(y, [0, 1]).all()
    if shuffled:
        y[:] = 0; rng = np.random.default_rng(277)
        for ids in train.groupby('k', sort=True).indices.values():
            y[rng.choice(ids)] = 1
    return y


def inputs(train, current, F):
    W.assert_pre_weekend(F)
    assert len(train) and len(current) and current.k.nunique() == current.date.nunique() == 1
    assert train.k.max() < current.k.min() and train.date.notna().all()
    assert train.date.lt(cutoff(current.date.iloc[0])).all()
    assert not train.duplicated(['k', 'driver']).any() and not current.driver.duplicated().any()
    train = train.sort_values(['k', 'driver']).reset_index(drop=True)
    pipe = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler())
    A = pipe.fit_transform(train[F]); B = pipe.transform(current[F])
    assert A.shape[1] == B.shape[1] == len(F) and np.isfinite(A).all() and np.isfinite(B).all()
    return train, A, B


def fit_score(train, current, F, context=True, shuffled=False):
    train, A, B = inputs(train, current, F)
    group = train.groupby('k', sort=True).ngroup().to_numpy()
    events = train.drop_duplicates('k').k.to_numpy()
    weights = 2.**((events-events.max())/100); weights /= weights.sum()
    y = labels(train, shuffled)
    with threadpool_limits(limits=1):
        fit = minimize(objective, initial(len(F)), args=(A, group, y, weights, context), jac=True,
                       method='L-BFGS-B', options={k: CONFIG[k] for k in ['maxiter', 'maxfun', 'ftol', 'gtol']})
        utility, _ = forward(fit.x, B, np.zeros(len(B), dtype=int), context)
    assert fit.status in [0, 1] and np.isfinite(fit.fun) and np.isfinite(fit.jac).all()
    probability = softmax(utility)
    assert np.isfinite(probability).all() and (probability > 0).all() and np.isclose(probability.sum(), 1)
    receipt = dict(theta=fit.x.tolist(), status=int(fit.status), iterations=int(fit.nit), loss=float(fit.fun),
                   gradient_max=float(np.abs(fit.jac).max()), context=context, shuffled=shuffled)
    return -probability, receipt


def main():
    assert not (ROOT / 'results/e277_inner_predictions.csv').exists(), 'do not overwrite scored experiment'
    results, pace = load_inputs(); X, F = make_features(results, pace)
    receipts = []
    def network(tr, te, context=True, shuffled=False):
        score, receipt = fit_score(tr, te, F, context, shuffled)
        receipts.append(dict(season=int(te.season.iloc[0]), round=int(te['round'].iloc[0]), **receipt))
        print('FIT', te.season.iloc[0], te['round'].iloc[0], context, shuffled, receipt['status'], receipt['iterations'], flush=True)
        return score
    for _, current in X[X.season.ge(2020)].groupby('k'):
        train = X[X.k.lt(current.k.iloc[0]) & X.season.ge(2019)]
        assert len(train) and train.date.lt(cutoff(current.date.iloc[0])).all()
    W.MODELS.update(
        standings=lambda tr, te, F: (-te.d_season_pts.fillna(0)*100+te.d10_order.fillna(11)*1e-3).to_numpy(),
        pre_logit=lambda tr, te, F: W.m_logit_pod(tr, te, F, C=.03, hl=100),
        win_logit=win_score,
        gbm_win=lambda tr, te, F: -lgb.LGBMClassifier(**W._lgb_params()).fit(tr[F], tr.win, sample_weight=W._w(tr, 100)).predict_proba(te[F])[:, 1],
        context_win=lambda tr, te, F: network(tr, te),
        row_win=lambda tr, te, F: network(tr, te, context=False),
        context_null=lambda tr, te, F: network(tr, te, shuffled=True))
    R, P = W.walk_forward(X, [2020, 2021, 2022], ['standings', 'pre_logit', 'win_logit', 'gbm_win', 'context_win', 'row_win', 'context_null'],
                          F=F, ens={'ens': ['standings', 'pre_logit']}, tie=None)
    assert len(R) == 61*8 and len(P) == 1220*8 and len(receipts) == 183
    truth = X[KEYS+['team', 'order', 'win']]; P = P.merge(truth, on=KEYS, validate='many_to_one')
    picked = P.sort_values(['score', 'driver']).drop_duplicates(['season', 'round', 'model'])
    winners = truth[truth.win.eq(1)].set_index(['season', 'round']).team
    models = {}; probabilities = []
    for (s, r, name), rows in P[~P.model.isin(['standings', 'ens'])].groupby(['season', 'round', 'model']):
        p = -rows.score.to_numpy(); p /= p.sum(); y = rows.win.to_numpy()
        probabilities.append(dict(season=int(s), round=int(r), model=name, logloss=float(-np.log(p[y == 1][0])), brier=float(np.sum((p-y)**2))))
    C = pd.DataFrame(probabilities)
    for name, rows in R.groupby('model'):
        chosen = picked[picked.model.eq(name)]
        models[name] = dict(winners=int(rows.winner.sum()), winner_rate=float(rows.winner.mean()), overlap=float(rows.podium.mean()),
                            by_season=rows.groupby('season').winner.sum().astype(int).to_dict(),
                            team_hits=sum(t == winners.loc[(s, r)] for s, r, t in chosen[['season', 'round', 'team']].itertuples(index=False, name=None)))
        if name not in ['standings', 'ens']: models[name].update(C[C.model.eq(name)][['logloss', 'brier']].mean().to_dict())
    wide = R.pivot(index=['season', 'round'], columns='model', values='winner').sort_index(); paired = {}
    for name in CONTROLS:
        diff, ci, p = W.paired_boot(wide.context_win, wide[name], n=5000, seed=0)
        paired[name] = dict(diff=float(diff), ci=ci.tolist(), p_nonpositive=float(p))
    summary = dict(candidate='context_win', features=F, config=CONFIG, models=models, paired=paired,
                   gate=bool(models['context_win']['winners'] >= 37 and all(v['ci'][0] > 0 for v in paired.values())
                             and models['context_win']['winners'] > models['context_null']['winners']),
                   evaluated_races=61, forecast_entrants=1220, outer_evaluated=False, evaluated_2026=False,
                   historical_roster_publication_verified=False,
                   optimizer_status_counts=pd.Series([r['status'] for r in receipts]).value_counts().to_dict(),
                   code_hashes={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                                ['experiments/e277_context_winner.py', 'experiments/e168_winner.py', 'f1/features.py', 'f1/walkforward.py']},
                   input_hashes={name: hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest() for name, frame in [('results_through_2022', results), ('pace_through_2022', pace)]})
    for suffix, frame in [('per_race', R), ('predictions', P), ('probability_metrics', C)]:
        frame.to_csv(ROOT / f'results/e277_inner_{suffix}.csv', index=False)
    (ROOT / 'results/e277_fit_receipts.json').write_text(json.dumps(receipts)+'\n')
    summary['artifact_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'results').glob('e277_*') if p.suffix in ['.csv', '.json']}
    (ROOT / 'results/e277_inner_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('RESULT', json.dumps(dict(models=models, paired=paired, gate=summary['gate'])), flush=True)


if __name__ == '__main__':
    main()
