"""Independent E271 official-label rankings/probabilities/bootstrap and direct fit."""
import hashlib
import json
import math

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.e271_supplier_winner import load_inputs, make_features, source_rows
from f1.data import ROOT


def main():
    directory = ROOT / 'results'
    summary = json.loads((directory / 'e271_inner_summary.json').read_text())
    P = pd.read_csv(directory / 'e271_inner_predictions.csv', float_precision='round_trip')
    R = pd.read_csv(directory / 'e271_inner_per_race.csv', float_precision='round_trip')
    C = pd.read_csv(directory / 'e271_inner_probability_metrics.csv', float_precision='round_trip')
    results, pace, projections, identities = load_inputs()
    truth = results[results.season.between(2020, 2022)].set_index(['season', 'round', 'driver'])
    assert len(P) == 1220*7 and len(R) == 61*7 and len(C) == 61*5
    assert not P.duplicated(['season', 'round', 'driver', 'model']).any()
    assert P.groupby('model').size().eq(1220).all() and R.groupby('model').size().eq(61).all()
    assert summary['evaluated_races'] == 61 and summary['forecast_entrants'] == 1220
    assert set(summary['paired']) == {'legacy_win', 'win_logit', 'standings', 'ens'}
    assert not any(summary[c] for c in ['outer_evaluated', 'evaluated_2026', 'historical_publication_time_verified', 'physical_oem_verified'])
    hits = {}; teams = {}; seasons = {}; losses = {}; overlaps = {}; chosen = {}
    for (year, rnd, name), group in P.groupby(['season', 'round', 'model'], sort=True):
        official = truth.loc[(year, rnd)]
        assert len(group) == len(official) == 20 and set(group.driver) == set(official.index)
        assert np.isfinite(group.score).all()
        predicted = [d for s, d in sorted(zip(group.score, group.driver))]
        actual = list(official.sort_values('order').index)
        hit = int(predicted[0] == actual[0]); overlap = len(set(predicted[:3]) & set(actual[:3]))/3
        saved = R[R.season.eq(year) & R['round'].eq(rnd) & R.model.eq(name)].iloc[0]
        assert saved.winner == hit
        np.testing.assert_allclose(saved.podium, overlap, atol=1e-15)
        assert saved.exact_podium == (set(predicted[:3]) == set(actual[:3]))
        for n in [5, 10]:
            np.testing.assert_allclose(saved[f'top{n}'], len(set(predicted[:n]) & set(actual[:n]))/n, atol=1e-15)
        np.testing.assert_allclose(saved.mae, sum(abs(i+1-official.loc[d, 'order']) for i, d in enumerate(predicted))/20, atol=1e-14)
        gain = {d: 3-i for i, d in enumerate(actual[:3])}
        dcg = sum(gain.get(d, 0)/math.log2(i+2) for i, d in enumerate(predicted[:3]))
        np.testing.assert_allclose(saved.ndcg3, dcg/sum((3-i)/math.log2(i+2) for i in range(3)), atol=1e-14)
        hits.setdefault(name, []).append(hit); overlaps.setdefault(name, []).append(overlap)
        chosen.setdefault(name, []).append(predicted[0])
        teams[name] = teams.get(name, 0)+int(official.loc[predicted[0], 'team'] == official.loc[actual[0], 'team'])
        seasons[name, year] = seasons.get((name, year), 0)+hit
        for row in group.itertuples():
            assert row.order == official.loc[row.driver, 'order'] and row.win == int(row.order == 1)
        if name not in ['standings', 'ens']:
            raw = [-float(s) for s in group.score]; assert all(0 < p < 1 for p in raw)
            p = [value/math.fsum(raw) for value in raw]
            loss = -math.log(p[list(group.driver).index(actual[0])])
            brier = math.fsum((v-int(d == actual[0]))**2 for d, v in zip(group.driver, p))
            stored = C[C.season.eq(year) & C['round'].eq(rnd) & C.model.eq(name)].iloc[0]
            np.testing.assert_allclose([stored.logloss, stored.brier], [loss, brier], atol=1e-13, rtol=1e-13)
            losses.setdefault(name, []).append([loss, brier])
    for name, values in hits.items():
        got = summary['models'][name]
        assert got['winners'] == sum(values) and got['winner_rate'] == sum(values)/61 and got['team_hits'] == teams[name]
        assert got['by_season'] == {str(year): seasons[name, year] for year in [2020, 2021, 2022]}
        np.testing.assert_allclose(got['overlap'], np.mean(overlaps[name]), atol=1e-15)
        if name in losses:
            np.testing.assert_allclose([got['logloss'], got['brier']], np.mean(losses[name], axis=0), atol=1e-13, rtol=1e-13)
    for name, pair in summary['paired'].items():
        d = np.array(hits['supplier_win'])-np.array(hits[name])
        indices = np.random.default_rng(0).integers(0, 61, size=(5000, 61))
        draws = d[indices].mean(axis=1)
        assert pair['diff'] == d.mean() and pair['p_nonpositive'] == (draws <= 0).mean()
        np.testing.assert_allclose(pair['ci'], np.quantile(draws, [.025, .975]), atol=1e-15, rtol=0)
        assert summary['winner_changes'][name] == sum(a != b for a, b in zip(chosen['supplier_win'], chosen[name]))
    gate = sum(hits['supplier_win']) >= 37 and all(v['ci'][0] > 0 for v in summary['paired'].values()) and sum(hits['supplier_win']) > sum(hits['supplier_null'])
    assert gate == summary['gate'] and summary['candidate'] == 'supplier_win'
    old = pd.read_csv(directory / 'e268_inner_predictions.csv', float_precision='round_trip')
    for name in ['standings', 'ens', 'win_logit', 'pre_logit']:
        a = P[P.model.eq(name)].sort_values(['season', 'round', 'driver'])
        b = old[old.model.eq(name)].sort_values(['season', 'round', 'driver'])
        np.testing.assert_array_equal(a[['season', 'round', 'driver']], b[['season', 'round', 'driver']])
        np.testing.assert_allclose(a.score, b.score, atol=1e-14, rtol=0)
    source = source_rows(results, projections, identities)
    X, base, _ = make_features(results, pace, source)
    for name, data in [('results_through_2022', results), ('pace_through_2022', pace), ('supplier_source', source)]:
        assert hashlib.sha256(data.to_csv(index=False).encode()).hexdigest() == summary['input_hashes'][name]
    assert summary['base_features'] == base and summary['candidate_features'] == base+['supplier_peer_points']
    assert summary['legacy_features'] == base+['legacy_peer_points']
    current = X[X.season.eq(2022) & X['round'].eq(10)]
    train = X[X.k.lt(current.k.iloc[0]) & X.season.ge(2019)]
    for name, F in [('supplier_win', summary['candidate_features']), ('legacy_win', summary['legacy_features'])]:
        model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=.03, max_iter=500))
        model.fit(train[F], train.win, logisticregression__sample_weight=2.**((train.k.to_numpy()-train.k.max())/100))
        direct = -model.predict_proba(current[F])[:, 1]
        stored = P[P.model.eq(name) & P.season.eq(2022) & P['round'].eq(10)].set_index('driver')
        np.testing.assert_allclose(direct, stored.loc[current.driver, 'score'], atol=1e-14, rtol=0)
    unknown = X[X.season.ge(2020)].assign(supplier_unknown=lambda d: d.supplier_brand.isna(), legacy_unknown=lambda d: d.legacy_brand.isna())
    for name, keys in [('unknown_by_event', ['season', 'round']), ('unknown_by_season', ['season'])]:
        expected = unknown.groupby(keys)[['supplier_unknown', 'legacy_unknown']].mean().reset_index().to_dict('records')
        assert summary[name] == expected
    for name, digest in summary['artifact_hashes'].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    print('E271 independent ALL61 official rankings/team/season/probabilities/four bootstraps/gate/references/hashes and direct fixed-target fits PASS')


if __name__ == '__main__':
    main()
