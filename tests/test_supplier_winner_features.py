"""E271 independent scalar histories and real raw-source/fitted timing checks."""
import copy
import json
import math

import numpy as np
import pandas as pd
import yaml

from experiments.e270_supplier_records import CACHE, LEGACY, project, verified
from experiments.e271_supplier_winner import NEW, add_history, load_inputs, make_features, score, source_rows
from f1.data import ROOT
from f1.profile import PU


def main():
    results, pace, projections, identities = load_inputs()
    source = source_rows(results, projections, identities)
    X, base, teams = make_features(results, pace, source)
    matched = pd.read_csv(ROOT / 'results/e270_matched.csv', keep_default_na=False)
    actual_source = dict(zip(map(tuple, source[['season', 'round', 'code']].to_numpy()), source.brand))
    for row in matched.itertuples():
        assert (actual_source.get((row.season, row.round, row.code)) or '') == (row.engineManufacturerId if row.covered else '')
    # Build source teams independently from official records and E270's audited join.
    brands = {(r.season, r.round, r.code): r.engineManufacturerId for r in matched.itertuples() if r.covered}
    raw = list(results.itertuples(index=False)); scalar = []
    event_keys = sorted({(r.season, r.round) for r in raw})
    event_k = {key: k for k, key in enumerate(event_keys)}
    for year, rnd in event_keys:
        rows = [r for r in raw if (r.season, r.round) == (year, rnd)]
        center = math.fsum(r.points/25 for r in rows)/len(rows)
        for kind in ['supplier', 'legacy']:
            groups = {}
            for r in rows:
                old = next((m.get(r.team) for (a, b), m in PU.items() if a <= year <= b), None)
                brand = brands.get((year, rnd, r.code)) if kind == 'supplier' else LEGACY.get(old)
                if brand:
                    groups.setdefault((r.team, brand), []).append(r.points/25)
            for (team, brand), points in groups.items():
                scalar.append(dict(kind=kind, season=year, round=rnd, k=event_k[year, rnd],
                                   date=rows[0].date, team=team, brand=brand, z=math.fsum(points)/len(points)-center))
    indexed = {(r.kind, r.season, r.round, r.team): r for r in teams.itertuples()}
    assert len(scalar) == len(teams)
    for r in scalar:
        got = indexed[r['kind'], r['season'], r['round'], r['team']]
        assert got.brand == r['brand'] and got.k == r['k']
        np.testing.assert_allclose(got.z, r['z'], atol=1e-15, rtol=1e-13)
    for r in X.itertuples():
        cutoff = r.date.normalize()-pd.Timedelta(days=(r.date.weekday()-3) % 7)
        for kind, feature in zip(['supplier', 'legacy'], NEW):
            past = [p for p in scalar if p['kind'] == kind and p['season'] == r.season and p['k'] < r.k and p['date'] < cutoff]
            own = [p for p in past if p['team'] == r.team]
            brand = max(own, key=lambda p: p['k'])['brand'] if own else None
            peers = [(2**(-(r.k-p['k'])/10), p['z']) for p in past if p['brand'] == brand and p['team'] != r.team] if brand else []
            expected = math.fsum(w*z for w, z in peers)/(5+math.fsum(w for w, z in peers))
            assert getattr(r, kind+'_brand') == brand
            np.testing.assert_allclose(getattr(r, feature), expected, atol=1e-14, rtol=1e-12)
    target = X.season.eq(2022) & X['round'].eq(10); k = X.loc[target, 'k'].iloc[0]
    changed_results = results.copy()
    future = changed_results.season.eq(2022) & changed_results['round'].ge(10)
    for col in ['grid', 'q_pos', 's_grid', 's_order', 'order', 'finish', 'points', 'laps', 'q1', 'q2', 'q3']:
        changed_results.loc[future, col] = 999
    changed_results.loc[future, 'status'] = 'poison'
    later = changed_results.season.eq(2022) & changed_results['round'].gt(10)
    for col in ['driver', 'team', 'circuit']:
        changed_results.loc[later, col] = changed_results.loc[later, col].astype(str)+'futureΩ'
    changed_pace = pace.copy()
    changed_pace.loc[changed_pace.season.eq(2022) & changed_pace['round'].ge(10), 'pace_gap'] = 999.
    # Reparse ACTUAL hash-verified source YAML after poisoning current/future bodies.
    poisoned = copy.deepcopy(projections)
    for event in poisoned:
        if (event['season'], event['round']) < (2022, 10):
            continue
        raw_bytes = (CACHE / (event['git_blob']+'.yml')).read_bytes()
        raw = yaml.safe_load(verified(raw_bytes, event['git_blob']))
        numbers = [r['driverNumber'] for r in raw][::-1]
        for row, number in zip(raw, numbers):
            for field in list(row):
                if field not in ['driverId', 'driverNumber', 'constructorId', 'engineManufacturerId', 'engineId']:
                    row[field] = {'poisoned_outcome_grid_text': [999]}
            row.update(driverId='future-identity', driverNumber=number,
                       constructorId='future-constructor', engineManufacturerId='future-brand', engineId='future-engine')
        event['rows'] = project(yaml.safe_dump(raw))
    changed_source = source_rows(changed_results, poisoned, identities)
    Z, changed_base, _ = make_features(changed_results, changed_pace, changed_source)
    assert base == changed_base
    pd.testing.assert_frame_equal(X.loc[X.k.le(k), base+NEW], Z.loc[Z.k.le(k), base+NEW])
    prefix_results = results[results.season.lt(2022) | (results.season.eq(2022) & results['round'].le(10))]
    prefix_pace = pace[pace.season.lt(2022) | (pace.season.eq(2022) & pace['round'].le(10))]
    past_projections = [r for r in projections if (r['season'], r['round']) < (2022, 10)]
    past_identities = [r for r in identities if (r['season'], r['round']) < (2022, 10)]
    prefix_source = source_rows(prefix_results, past_projections, past_identities)
    prefix, _, _ = make_features(prefix_results, prefix_pace, prefix_source)
    pd.testing.assert_frame_equal(X.loc[X.k.le(k), base+NEW], prefix[base+NEW])
    train = X.k.lt(k) & X.season.ge(2019)
    scores = []
    for features, shuffled in [(base+[NEW[0]], False), (base+[NEW[1]], False), (base+[NEW[0]], True)]:
        a = score(X[train], X[target], features, shuffled)
        b = score(Z[train], Z[target], features, shuffled)
        c = score(prefix[train.loc[prefix.index]], prefix[target.loc[prefix.index]], features, shuffled)
        np.testing.assert_array_equal(a, b); np.testing.assert_array_equal(a, c)
        scores.append(a)
    # Actual earlier supplier records must affect a supported later peer history.
    earlier = copy.deepcopy(projections)
    event = next(r for r in earlier if (r['season'], r['round']) == (2022, 9))
    cache_bytes = verified((CACHE / (event['git_blob']+'.yml')).read_bytes(), event['git_blob'])
    raw = yaml.safe_load(cache_bytes)
    selected = [r for r in raw if r['constructorId'] == 'williams']
    assert len(selected) == 2
    for r in selected:
        r['engineManufacturerId'] = 'earlier-independent-brand'
    event['rows'] = project(yaml.safe_dump(raw))
    response_source = source_rows(results, earlier, identities)
    responsive, _, _ = make_features(results, pace, response_source)
    assert not X.loc[target, NEW[0]].equals(responsive.loc[target, NEW[0]])
    assert not np.array_equal(scores[0], score(responsive[train], responsive[target], base+[NEW[0]]))
    earlier_points = results.copy()
    earlier_points.loc[results.season.eq(2022) & results['round'].eq(9) & results.team.eq('williams'), 'points'] += 25
    points_response, _, _ = make_features(earlier_points, pace, source)
    assert not X.loc[target, NEW[0]].equals(points_response.loc[target, NEW[0]])
    # Independent null labels and fit, without inspecting target outcome accuracy.
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    training = X[train]; rng = np.random.default_rng(271); y = np.zeros(len(training), int)
    prefix_labels = None
    prefix_last = sorted(training.k.unique())[9]
    for event in sorted(training.k.unique()):
        positions = np.flatnonzero(training.k.to_numpy() == event)
        y[positions[rng.integers(len(positions))]] = 1
        if event == prefix_last:
            prefix_labels = y[:positions[-1]+1].copy()
    assert y.sum() == training.k.nunique() and prefix_labels is not None
    rng = np.random.default_rng(271); prefix_y = np.zeros(len(prefix_labels), int)
    for event in sorted(training.k.unique()):
        positions = np.flatnonzero(training.k.to_numpy()[:len(prefix_y)] == event)
        if len(positions):
            prefix_y[positions[rng.integers(len(positions))]] = 1
    np.testing.assert_array_equal(prefix_labels, prefix_y)
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=.03, max_iter=500))
    F = base+[NEW[0]]
    model.fit(training[F], y, logisticregression__sample_weight=2.**((training.k.to_numpy()-training.k.max())/100))
    np.testing.assert_array_equal(-model.predict_proba(X.loc[target, F])[:, 1], scores[2])
    saved_path = ROOT / 'results/e271_features.csv'
    if saved_path.exists():
        saved = pd.read_csv(saved_path, float_precision='round_trip')
        np.testing.assert_array_equal(saved[['season', 'round', 'code', 'k']], X[['season', 'round', 'code', 'k']])
        np.testing.assert_array_equal(saved[NEW], X[NEW])
        predictions = pd.read_csv(ROOT / 'results/e271_inner_predictions.csv', float_precision='round_trip')
        for name, fitted in zip(['supplier_win', 'legacy_win', 'supplier_null'], scores):
            stored = predictions[predictions.season.eq(2022) & predictions['round'].eq(10) & predictions.model.eq(name)].set_index('driver')
            np.testing.assert_allclose(stored.loc[X.loc[target, 'driver'], 'score'], fitted, atol=1e-14, rtol=0)
    print('E271 independent ALL source-team/2060 scalar histories, actual raw current/future poison+prefix, earlier supplier/points/fitted response, direct null PASS')


if __name__ == '__main__':
    main()
