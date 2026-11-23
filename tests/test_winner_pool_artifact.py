"""Independent scalar E278 official-label / fixed prediction-pool audit."""
import csv
import hashlib
import json
from collections import Counter

import numpy as np

from f1.data import DATA, ROOT


def main():
    directory = ROOT / 'results'
    s = json.loads((directory / 'e278_winner_pool_summary.json').read_text())
    rows = json.loads((directory / 'e278_winner_pool_events.json').read_text())
    names = ['standings', 'ens', 'win_logit', 'gbm_base', 'ordered_ids']
    assert s['pool'] == names and len(rows) == s['races'] == 61 and s['entrants'] == 1220
    assert s['diagnostic_only'] and s['source_immutable']
    assert not any(s[k] for k in ['trained_selector', 'outer_evaluated', 'evaluated_2026', 'universal_accuracy_ceiling'])
    official = {}
    with (DATA / 'results.csv').open() as file:
        reader = csv.reader(file); header = next(reader); ix = {n: header.index(n) for n in ['season', 'round', 'driver', 'team', 'order']}
        for fields in reader:
            year = int(fields[ix['season']])
            if year < 2020 or year > 2022: continue
            key = year, int(fields[ix['round']]), fields[ix['driver']]
            assert key not in official
            official[key] = (fields[ix['team']], int(fields[ix['order']]))
    prediction = {}; all_rows = []
    with (directory / 'e268_inner_predictions.csv').open() as file:
        for row in csv.DictReader(file):
            year = int(row['season']); assert year in [2020, 2021, 2022]
            if row['model'] not in names: continue
            key = year, int(row['round']), row['driver'], row['model']
            assert key not in prediction
            prediction[key] = float(row['score']); all_rows.append(key)
    assert len(prediction) == 6100 and all(np.isfinite(v) for v in prediction.values())
    assert len(official) == 1220 and len(set((y, r) for y, r, d in official)) == 61
    hits = {n: [] for n in names}; overlap = {n: [] for n in names}; team_hits = Counter()
    oracle = []; partitions = Counter(); misses = Counter(); years = {}
    for saved in rows:
        y, r = saved['season'], saved['round']
        entries = [(d, t, p) for (sy, sr, d), (t, p) in official.items() if (sy, sr) == (y, r)]
        actual = [d for d, t, p in sorted(entries, key=lambda row: row[2])]
        assert len(actual) == 20 and sorted(p for d, t, p in entries) == list(range(1, 21))
        winning_team = next(t for d, t, p in entries if p == 1)
        assert saved['winner'] == actual[0] and saved['winning_team'] == winning_team
        selected = []; selected_teams = []
        for name in names:
            order = sorted((prediction[y, r, d, name], d) for d, t, p in entries)
            predicted = [d for score, d in order]; selected.append(predicted[0])
            team = official[y, r, predicted[0]][0]; selected_teams.append(team)
            assert saved['picks'][name] == predicted[0]
            hits[name].append(int(predicted[0] == actual[0])); team_hits[name] += int(team == winning_team)
            overlap[name].append(sum(d in actual[:3] for d in predicted[:3])/3)
        assert saved['picked_teams'] == sorted(set(selected_teams))
        assert saved['distinct_drivers'] == len(set(selected)) and saved['distinct_teams'] == len(set(selected_teams))
        good = int(any(d == actual[0] for d in selected)); oracle.append(good)
        assert saved['oracle_hit'] == good
        partition = ('unanimous_' if len(set(selected)) == 1 else 'disagreement_')+('correct' if good else 'wrong') if len(set(selected)) == 1 else ('disagreement_any_correct' if good else 'disagreement_none_correct')
        miss = None if good else 'winning_team_covered_wrong_driver' if winning_team in selected_teams else 'winning_team_absent'
        assert saved['partition'] == partition and saved['miss_type'] == miss
        partitions[partition] += 1
        if miss: misses[miss] += 1
        state = years.setdefault(str(y), dict(races=0, oracle_winners=0, partitions=Counter(), misses=Counter()))
        state['races'] += 1; state['oracle_winners'] += good; state['partitions'][partition] += 1
        if miss: state['misses'][miss] += 1
    assert dict(partitions) == s['partitions'] and dict(misses) == s['misses'] and years == s['by_season']
    assert sum(oracle) == s['oracle_winners'] and sum(oracle)/61 == s['oracle_rate']
    assert s['numerical_75_count'] == 46 and s['pool_reaches_numerical_75'] == (sum(oracle) >= 46)
    assert sum(s['misses'].values()) == 61-sum(oracle)
    for n in names:
        got = s['models'][n]
        assert got['winners'] == sum(hits[n]) and got['winner_rate'] == sum(hits[n])/61 and got['team_hits'] == team_hits[n]
        np.testing.assert_allclose(got['strict_podium_overlap'], np.mean(overlap[n]), atol=1e-15)
    assert set(s['hindsight_paired_headroom']) == {'standings', 'ens', 'win_logit'}
    for n, pair in s['hindsight_paired_headroom'].items():
        d = np.array(oracle)-np.array(hits[n]); indices = np.random.default_rng(0).integers(0, 61, size=(5000, 61))
        draws = d[indices].mean(axis=1)
        assert pair['diff'] == d.mean() and pair['p_nonpositive'] == (draws <= 0).mean()
        np.testing.assert_allclose(pair['ci'], np.quantile(draws, [.025, .975]), atol=1e-15, rtol=0)
    with (directory / 'e268_inner_per_race.csv').open() as file:
        for old in csv.DictReader(file):
            n = old['model']
            if n not in names: continue
            idx = next(i for i, row in enumerate(rows) if (row['season'], row['round']) == (int(old['season']), int(old['round'])))
            assert hits[n][idx] == float(old['winner'])
            np.testing.assert_allclose(overlap[n][idx], float(old['podium']), atol=1e-15)
    for path in [DATA / 'results.csv', directory / 'e268_inner_predictions.csv']:
        assert s['source_hashes'][path.name] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert s['code_sha256'] == hashlib.sha256((ROOT / 'experiments/e278_winner_pool.py').read_bytes()).hexdigest()
    assert s['events_sha256'] == hashlib.sha256((directory / 'e278_winner_pool_events.json').read_bytes()).hexdigest()
    assert len(set((row['season'], row['round']) for row in rows)) == 61
    print('E278 independent ALL305 rankings/61 official cohorts/partitions/season totals/hindsight 3bootstraps/count flags/frozen reference metrics/hashes PASS')


if __name__ == '__main__':
    main()
