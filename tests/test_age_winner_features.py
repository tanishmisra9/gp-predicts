"""E275 independent all-row history math and actual raw-source/fitted guards."""
import copy
import datetime as dt
import hashlib
import json
import sqlite3
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.e274_birth_source import CACHE, project
from experiments.e275_age_winner import AGES, EXPERIENCE, KEYS, NEW, load_inputs, make_features, score
from f1.data import ROOT, _s


def main():
    results, pace, source = load_inputs(); X, base = make_features(results, pace, source)
    raw_entries = list(results[['season', 'round', 'driver', 'date']].itertuples(index=False, name=None))
    for row in X.itertuples():
        date = row.date.date(); cutoff = date-dt.timedelta(days=(date.isoweekday()-4) % 7)
        earlier = [(s, r, d, t) for s, r, d, t in raw_entries if d == row.driver
                   and (s, r) < (row.season, row.round) and t.date() < cutoff]
        assert row.observed_entry_count == len(earlier) and row.observed_rookie_flag == (len(earlier) < 10)
        births = {r['dateOfBirth'] for r in source if r['driver'] == row.driver and r['dateOfBirth'] is not None
                  and (r['season'], r['round']) < (row.season, row.round)
                  and dt.datetime.strptime(r['date'], '%Y-%m-%d').date() < cutoff}
        if len(births) == 1:
            dob = sorted(births)[0]; birth = dt.datetime.strptime(dob, '%Y-%m-%d').date()
            expected = (cutoff-birth).days/365.2425
            assert row.prior_dob == dob and row.prior_age_years == expected
            np.testing.assert_allclose(row.prior_age_squared, expected**2, atol=1e-12, rtol=1e-15)
        else:
            assert row.prior_dob is None and np.isnan(row.prior_age_years) and np.isnan(row.prior_age_squared)
    target = X.season.eq(2022) & X['round'].eq(10); k = X.loc[target, 'k'].iloc[0]
    future = results.season.eq(2022) & results['round'].ge(10)
    later = results.season.eq(2022) & results['round'].gt(10)
    changed_results = results.copy()
    for col in ['order', 'finish', 'points', 'laps', 'grid', 'q_pos', 'q1', 'q2', 'q3', 's_grid', 's_order', 's_points']:
        changed_results.loc[future, col] = 999
    changed_results.loc[future, 'status'] = 'poison'
    for col in ['driver', 'team', 'circuit']:
        changed_results.loc[later, col] = changed_results.loc[later, col].astype(str)+'futureΩ'
    changed_pace = pace.copy()
    changed_pace.loc[changed_pace.season.eq(2022) & changed_pace['round'].ge(10), 'pace_gap'] = 999.
    mapping = {(r.season, r.round, r.driver): r.date.strftime('%Y-%m-%d') for r in changed_results.itertuples()}
    source_report = json.loads((ROOT / 'results/e274_birth_source_summary.json').read_text())
    cache_before = hashlib.sha256(CACHE.read_bytes()).hexdigest()
    raw_pages = []; poisoned = []
    with sqlite3.connect(f'file:{CACHE}?mode=ro', uri=True) as connection, \
            patch('requests.adapters.HTTPAdapter.send', side_effect=AssertionError('raw mutation HTTP forbidden')) as sends:
        for receipt in source_report['pages']:
            blob = connection.execute('SELECT value FROM responses WHERE key=?', (receipt['cache_key'],)).fetchone()[0]
            assert hashlib.sha256(blob).hexdigest() == receipt['blob_sha256']
            response = _s.cache.responses.deserialize(receipt['cache_key'], blob)
            assert hashlib.sha256(response.content).hexdigest() == receipt['body_sha256']
            races = json.loads(response.content)['MRData']['RaceTable']['Races']
            raw_pages.append(races)
            changed = copy.deepcopy(races)
            for race in changed:
                event = int(race['season']), int(race['round'])
                if event < (2022, 10):
                    continue
                names = [r['Driver']['driverId'] for r in race['Results']][::-1]
                for result, identity in zip(race['Results'], names):
                    for field in list(result):
                        if field != 'Driver':
                            result[field] = {'outcome_grid_text_poison': [999]}
                    for field in list(result['Driver']):
                        if field not in ['driverId', 'dateOfBirth']:
                            result['Driver'][field] = {'identity_text_poison': [999]}
                    result['Driver']['dateOfBirth'] = '1999-01-01'
                    result['Driver']['driverId'] = identity+('futureΩ' if event > (2022, 10) else '')
            poisoned.extend(project(changed, mapping))
        assert sends.call_count == 0
    assert hashlib.sha256(CACHE.read_bytes()).hexdigest() == cache_before
    Z, changed_base = make_features(changed_results, changed_pace, poisoned)
    assert base == changed_base
    pd.testing.assert_frame_equal(X.loc[X.k.le(k), base+NEW], Z.loc[Z.k.le(k), base+NEW])
    mask = results.season.lt(2022) | (results.season.eq(2022) & results['round'].le(10))
    pmask = pace.season.lt(2022) | (pace.season.eq(2022) & pace['round'].le(10))
    prefix_source = [r for r in source if (r['season'], r['round']) < (2022, 10)]
    prefix, _ = make_features(results[mask], pace[pmask], prefix_source)
    pd.testing.assert_frame_equal(X.loc[X.k.le(k), base+NEW], prefix[base+NEW])
    train = X.k.lt(k) & X.season.ge(2019); scores = []
    for features, shuffled in [(base+NEW, False), (base+EXPERIENCE, False), (base+NEW, True)]:
        a = score(X[train], X[target], features, shuffled)
        np.testing.assert_array_equal(a, score(Z[train], Z[target], features, shuffled))
        np.testing.assert_array_equal(a, score(prefix[prefix.k.lt(k) & prefix.season.ge(2019)], prefix[prefix.k.eq(k)], features, shuffled))
        scores.append(a)
    # Coherent earlier raw DOB edits must change supported later age and fitted scores.
    selected = sorted(X.loc[target, 'driver'])[0]; changed_source = []
    original_map = {(r.season, r.round, r.driver): r.date.strftime('%Y-%m-%d') for r in results.itertuples()}
    for races in raw_pages:
        earlier = copy.deepcopy(races)
        for race in earlier:
            if (int(race['season']), int(race['round'])) >= (2022, 10):
                continue
            for result in race['Results']:
                if result['Driver']['driverId'] == selected:
                    birth = dt.datetime.strptime(result['Driver']['dateOfBirth'], '%Y-%m-%d').date()+dt.timedelta(days=365)
                    result['Driver']['dateOfBirth'] = birth.isoformat()
        changed_source.extend(project(earlier, original_map))
    response, _ = make_features(results, pace, changed_source)
    assert not response.loc[target, AGES].equals(X.loc[target, AGES])
    assert not np.array_equal(scores[0], score(response[train], response[target], base+NEW))
    # Earlier presence changes the left-censored count; labels alone do not.
    earliest = results[results.driver.eq(selected)].sort_values(['season', 'round']).iloc[0]
    drop = results.season.eq(earliest.season) & results['round'].eq(earliest['round']) & results.driver.eq(selected)
    fewer_source = [r for r in source if (r['season'], r['round'], r['driver']) != (earliest.season, earliest['round'], selected)]
    fewer_pace = pace[~(pace.season.eq(earliest.season) & pace['round'].eq(earliest['round']) & pace.code.eq(earliest.code))]
    fewer, _ = make_features(results[~drop], fewer_pace, fewer_source)
    assert fewer[fewer.season.eq(2022) & fewer['round'].eq(10) & fewer.driver.eq(selected)].observed_entry_count.iloc[0] == X.loc[target & X.driver.eq(selected), EXPERIENCE[0]].iloc[0]-1
    labels_only = results.copy(); labels_only['order'] = 21-labels_only.order
    labels_only['points'] += 100; labels_only['grid'] = 999
    label_features, _ = make_features(labels_only, pace, source)
    pd.testing.assert_frame_equal(X[NEW], label_features[NEW])
    # Regenerate the prefix-stable null and fit its frozen sklearn specification independently.
    training = X[train]; y = np.zeros(len(training), int); rng = np.random.default_rng(275)
    for event in sorted(training.k.unique()):
        positions = np.flatnonzero(training.k.to_numpy() == event)
        y[positions[rng.integers(len(positions))]] = 1
    assert y.sum() == training.k.nunique()
    length = np.flatnonzero(training.k.to_numpy() <= sorted(training.k.unique())[9])[-1]+1
    prefix_y = np.zeros(length, int); rng = np.random.default_rng(275)
    for event in sorted(set(training.k.to_numpy()[:length])):
        positions = np.flatnonzero(training.k.to_numpy()[:length] == event)
        prefix_y[positions[rng.integers(len(positions))]] = 1
    np.testing.assert_array_equal(y[:length], prefix_y)
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=.03, max_iter=500))
    model.fit(training[base+NEW], y, logisticregression__sample_weight=2.**((training.k.to_numpy()-training.k.max())/100))
    np.testing.assert_allclose(-model.predict_proba(X.loc[target, base+NEW])[:, 1], scores[2], atol=1e-14, rtol=0)
    if (ROOT / 'results/e275_features.csv').exists():
        stored = pd.read_csv(ROOT / 'results/e275_features.csv', float_precision='round_trip')
        np.testing.assert_array_equal(stored[KEYS+['k']], X[KEYS+['k']])
        np.testing.assert_array_equal(stored[NEW], X[NEW])
        P = pd.read_csv(ROOT / 'results/e275_inner_predictions.csv', float_precision='round_trip')
        for name, fitted in zip(['age_win', 'experience_win', 'age_null'], scores):
            rows = P[P.season.eq(2022) & P['round'].eq(10) & P.model.eq(name)].set_index('driver')
            np.testing.assert_allclose(rows.loc[X.loc[target, 'driver'], 'score'], fitted, atol=1e-14, rtol=0)
    print('E275 independent ALL2060 age/count/DOBs, ALL24 actual raw future poison/prefix, earlier DOB/fitted/presence, labels/null/cache guards PASS')


if __name__ == '__main__':
    main()
