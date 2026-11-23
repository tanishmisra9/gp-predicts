"""Independent all-race E277 cohort, scoring, parameters, probability and gate audit."""
import hashlib
import json
import math

import numpy as np
import pandas as pd

from experiments.e277_context_winner import CONFIG, CONTROLS, fit_score, load_inputs, make_features
from f1.data import ROOT
from tests.test_context_winner import scalar_forward


def main():
    directory = ROOT / 'results'
    summary = json.loads((directory / 'e277_inner_summary.json').read_text())
    receipts = json.loads((directory / 'e277_fit_receipts.json').read_text())
    P = pd.read_csv(directory / 'e277_inner_predictions.csv', float_precision='round_trip')
    R = pd.read_csv(directory / 'e277_inner_per_race.csv', float_precision='round_trip')
    C = pd.read_csv(directory / 'e277_inner_probability_metrics.csv', float_precision='round_trip')
    results, pace = load_inputs(); X, F = make_features(results, pace)
    truth = results[results.season.ge(2020)].set_index(['season', 'round', 'driver'])
    assert len(P) == 1220*8 and len(R) == 61*8 and len(C) == 61*6 and len(receipts) == 183
    assert P.groupby('model').size().eq(1220).all() and R.groupby('model').size().eq(61).all()
    assert not P.duplicated(['season', 'round', 'driver', 'model']).any()
    assert summary['features'] == F and summary['config'] == CONFIG
    assert CONFIG == dict(width=8, l2=.01, half_life=100, init_seed=0, null_seed=277,
                          maxiter=300, maxfun=10000, ftol=1e-10, gtol=1e-6)
    assert summary['evaluated_races'] == 61 and summary['forecast_entrants'] == 1220
    assert not any(summary[n] for n in ['outer_evaluated', 'evaluated_2026', 'historical_roster_publication_verified'])
    assert set(summary['paired']) == set(CONTROLS)
    hits = {}; seasons = {}; teams = {}; overlaps = {}; losses = {}
    for (year, rnd, name), group in P.groupby(['season', 'round', 'model'], sort=True):
        official = truth.loc[(year, rnd)]
        assert len(group) == len(official) == 20 and set(group.driver) == set(official.index)
        assert np.isfinite(group.score).all()
        pred = [d for s, d in sorted(zip(group.score, group.driver))]
        actual = list(official.sort_values('order').index)
        hit = int(pred[0] == actual[0]); overlap = len(set(pred[:3]) & set(actual[:3]))/3
        saved = R[R.season.eq(year) & R['round'].eq(rnd) & R.model.eq(name)].iloc[0]
        assert saved.winner == hit and saved.exact_podium == (set(pred[:3]) == set(actual[:3]))
        np.testing.assert_allclose(saved.podium, overlap, atol=1e-15)
        for n in [5, 10]: np.testing.assert_allclose(saved[f'top{n}'], len(set(pred[:n]) & set(actual[:n]))/n, atol=1e-15)
        np.testing.assert_allclose(saved.mae, sum(abs(i+1-official.loc[d, 'order']) for i, d in enumerate(pred))/20, atol=1e-14)
        gains = {d: 3-i for i, d in enumerate(actual[:3])}
        dcg = sum(gains.get(d, 0)/math.log2(i+2) for i, d in enumerate(pred[:3]))
        np.testing.assert_allclose(saved.ndcg3, dcg/sum((3-i)/math.log2(i+2) for i in range(3)), atol=1e-14)
        hits.setdefault(name, []).append(hit); overlaps.setdefault(name, []).append(overlap)
        teams[name] = teams.get(name, 0)+int(official.loc[pred[0], 'team'] == official.loc[actual[0], 'team'])
        seasons[name, year] = seasons.get((name, year), 0)+hit
        for row in group.itertuples():
            assert row.order == official.loc[row.driver, 'order'] and row.win == int(row.order == 1)
        if name not in ['standings', 'ens']:
            raw = [-float(s) for s in group.score]; assert all(0 < p < 1 for p in raw)
            prob = [v/math.fsum(raw) for v in raw]
            logloss = -math.log(prob[list(group.driver).index(actual[0])])
            brier = math.fsum((v-int(d == actual[0]))**2 for v, d in zip(prob, group.driver))
            saved = C[C.season.eq(year) & C['round'].eq(rnd) & C.model.eq(name)].iloc[0]
            np.testing.assert_allclose([saved.logloss, saved.brier], [logloss, brier], atol=1e-13)
            losses.setdefault(name, []).append([logloss, brier])
    for name, values in hits.items():
        got = summary['models'][name]
        assert got['winners'] == sum(values) and got['winner_rate'] == sum(values)/61 and got['team_hits'] == teams[name]
        assert got['by_season'] == {str(y): seasons[name, y] for y in [2020, 2021, 2022]}
        np.testing.assert_allclose(got['overlap'], np.mean(overlaps[name]), atol=1e-15)
        if name in losses: np.testing.assert_allclose([got['logloss'], got['brier']], np.mean(losses[name], axis=0), atol=1e-13)
    for name, pair in summary['paired'].items():
        differences = np.array(hits['context_win'])-np.array(hits[name])
        indices = np.random.default_rng(0).integers(0, 61, size=(5000, 61)); draws = differences[indices].mean(axis=1)
        assert pair['diff'] == differences.mean() and pair['p_nonpositive'] == (draws <= 0).mean()
        np.testing.assert_allclose(pair['ci'], np.quantile(draws, [.025, .975]), atol=1e-15, rtol=0)
    gate = sum(hits['context_win']) >= 37 and all(v['ci'][0] > 0 for v in summary['paired'].values()) and sum(hits['context_win']) > sum(hits['context_null'])
    assert summary['gate'] == gate
    old = pd.read_csv(directory / 'e268_inner_predictions.csv', float_precision='round_trip')
    for name, oldname in [('standings', 'standings'), ('ens', 'ens'), ('win_logit', 'win_logit'), ('pre_logit', 'pre_logit'), ('gbm_win', 'gbm_base')]:
        a = P[P.model.eq(name)].sort_values(['season', 'round', 'driver']); b = old[old.model.eq(oldname)].sort_values(['season', 'round', 'driver'])
        np.testing.assert_array_equal(a[['season', 'round', 'driver']], b[['season', 'round', 'driver']])
        np.testing.assert_allclose(a.score, b.score, atol=1e-14, rtol=0)
    keys = set(); statuses = {}
    for receipt in receipts:
        year, rnd = receipt['season'], receipt['round']
        name = 'context_null' if receipt['shuffled'] else 'context_win' if receipt['context'] else 'row_win'
        assert (year, rnd, name) not in keys; keys.add((year, rnd, name))
        current = X[X.season.eq(year) & X['round'].eq(rnd)]
        train = X[X.k.lt(current.k.iloc[0]) & X.season.ge(2019)].sort_values(['k', 'driver'])
        assert len(train) and train.date.max() < current.date.iloc[0]-pd.Timedelta(days=3)
        values = train[F].to_numpy(); target = current[F].to_numpy()
        medians = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0 for col in values.T])
        imputed = np.where(np.isnan(values), medians, values); mean = imputed.mean(axis=0)
        scale = imputed.std(axis=0); scale[scale == 0] = 1
        B = (np.where(np.isnan(target), medians, target)-mean)/scale
        utility = scalar_forward(receipt['theta'], B, receipt['context'])
        prob = np.exp(utility-utility.max()); prob /= math.fsum(prob)
        saved = P[P.model.eq(name) & P.season.eq(year) & P['round'].eq(rnd)].set_index('driver')
        np.testing.assert_allclose(-prob, saved.loc[current.driver, 'score'], atol=1e-12, rtol=1e-12)
        assert receipt['status'] in [0, 1] and 0 <= receipt['iterations'] <= 300
        assert all(math.isfinite(receipt[field]) for field in ['loss', 'gradient_max'])
        assert receipt['loss'] >= 0 and receipt['gradient_max'] >= 0
        statuses[str(receipt['status'])] = statuses.get(str(receipt['status']), 0)+1
    assert statuses == summary['optimizer_status_counts']
    current = X[X.season.eq(2022) & X['round'].eq(10)]; train = X[X.k.lt(current.k.iloc[0]) & X.season.ge(2019)]
    for context, shuffled, name in [(True, False, 'context_win'), (False, False, 'row_win'), (True, True, 'context_null')]:
        direct, receipt = fit_score(train, current, F, context, shuffled)
        saved = P[P.model.eq(name) & P.season.eq(2022) & P['round'].eq(10)].set_index('driver')
        np.testing.assert_array_equal(direct, saved.loc[current.driver, 'score'])
        original = next(r for r in receipts if r['season'] == 2022 and r['round'] == 10 and r['context'] == context and r['shuffled'] == shuffled)
        assert receipt == {k: v for k, v in original.items() if k not in ['season', 'round']}
        # Independent actual training labels/null and scalar training objective.
        prior = train.sort_values(['k', 'driver']).reset_index(drop=True)
        values = prior[F].to_numpy()
        medians = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0 for col in values.T])
        imputed = np.where(np.isnan(values), medians, values); scale = imputed.std(axis=0); scale[scale == 0] = 1
        A = (imputed-imputed.mean(axis=0))/scale
        rng = np.random.default_rng(277); event_losses = []; event_weights = []
        for event, rows in prior.groupby('k', sort=True):
            indices = rows.index.to_numpy()
            chosen = int(rng.integers(len(indices))) if shuffled else int(np.flatnonzero(rows.win.to_numpy() == 1)[0])
            assert rows.win.sum() == 1
            utility = scalar_forward(receipt['theta'], A[indices], context)
            maximum = max(utility)
            event_losses.append(maximum+math.log(math.fsum(math.exp(float(u)-maximum) for u in utility))-utility[chosen])
            event_weights.append(2.**((event-prior.k.max())/100))
        loss = sum(w*l for w, l in zip(event_weights, event_losses))/math.fsum(event_weights)+.005*math.fsum(t*t for t in receipt['theta'])
        np.testing.assert_allclose(loss, receipt['loss'], atol=1e-12, rtol=1e-12)
    for name, frame in [('results_through_2022', results), ('pace_through_2022', pace)]:
        assert summary['input_hashes'][name] == hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    for name, digest in summary['artifact_hashes'].items(): assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    for name, digest in summary['code_hashes'].items(): assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    print('E277 independent ALL61 official cohorts/rankings/probability/team/season/5bootstraps/gate/frozen arrays, ALL183 scalar parameter forwards and 3 deterministic fixed-target refits PASS')


if __name__ == '__main__':
    main()
