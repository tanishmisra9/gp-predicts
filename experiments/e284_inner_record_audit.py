"""E284: independently reconstruct existing E185 prior rankings; no fitting."""
import csv
import hashlib
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MEASURES = ['med_gap', 'p25_gap', 'fcorr_gap', 'stint1_gap', 'best_stint_gap',
            'tcorr_gap', 'matched_gap', 'joint_gap', 'fresh_gap']
FILES = [('pacev_2018_2021.csv', dict(zip(MEASURES[:6], MEASURES[:6]))),
         ('pacev_2022_2024.csv', dict(zip(MEASURES[:6], MEASURES[:6]))),
         ('matchedpace_2018_2025.csv', {'matched_gap': 'matched_gap'}),
         ('jointpace_2018_2025.csv', {'joint_gap': 'joint_gap'}),
         ('jointpace_fresh_2018_2025.csv', {'joint_gap': 'fresh_gap'})]


def rows(path, first=2018, last=2022):
    with path.open() as file:
        # Do not interpret later-year numerical outcome/measurement fields.
        return [r for r in csv.DictReader(file) if first <= int(r['season']) <= last]


def calendar(entries):
    events = {}
    code_identity = {}
    for r in entries:
        assert code_identity.setdefault(r['code'], r['driver']) == r['driver']
        key = (int(r['season']), int(r['round']))
        day = date.fromisoformat(r['date'])
        assert day.year == key[0] and key[1] > 0
        event = events.setdefault(key, {'date': day, 'entries': []})
        assert event['date'] == day
        event['entries'].append(r)
    for k, key in enumerate(sorted(events)):
        event = events[key]
        event['k'] = k
        people = event['entries']
        for field in ('driver', 'code'):
            assert all(r[field] for r in people)
            assert len({r[field] for r in people}) == len(people)
        assert sorted(int(r['order']) for r in people) == list(range(1, len(people)+1))
        if k:
            assert events[sorted(events)[k-1]]['date'] < event['date']
    assert events
    return events


def sources(events):
    out = {m: {} for m in MEASURES}
    for filename, mapping in FILES:
        seen = set()
        for r in rows(ROOT/'data/parts'/filename):
            key = ((int(r['season']), int(r['round'])), r['code'])
            assert key not in seen
            seen.add(key)
            assert key[0] in events
            assert key[1] in {x['code'] for x in events[key[0]]['entries']}
            for column, measure in mapping.items():
                assert key not in out[measure]
                value = float(r[column]) if r[column] else math.nan
                assert math.isfinite(value) or math.isnan(value)
                out[measure][key] = value
    return out


def scores(events, source, target):
    """Finite-or-missing earlier observations -> scores; no history -> infinity."""
    cur = events[target]
    cutoff = cur['date']-timedelta(days=(cur['date'].weekday()-3) % 7)
    histories = {}
    for (key, code), value in sorted(source.items()):
        assert key in events and (math.isfinite(value) or math.isnan(value))
        if events[key]['k'] >= cur['k'] or math.isnan(value):
            continue
        assert events[key]['date'] < cutoff
        weight = .5**((cur['k']-events[key]['k'])/10)
        histories.setdefault(code, []).append((weight, value))
    result = {}
    for r in cur['entries']:
        observations = histories.get(r['code'], [])
        result[r['driver']] = (math.fsum(w*v for w, v in observations) /
                               math.fsum(w for w, v in observations)
                               if observations else math.inf)
    return result


def picks(values):
    return sorted(values, key=lambda driver: (values[driver], driver))[:3]


def metrics(event, chosen):
    truth = {r['driver']: r for r in event['entries']}
    assert len(chosen) == 3 and len(set(chosen)) == 3 and set(chosen) <= truth.keys()
    winner = next(r for r in event['entries'] if int(r['order']) == 1)
    return dict(winner=int(chosen[0] == winner['driver']),
                podium=len(set(chosen) & {r['driver'] for r in event['entries']
                                         if int(r['order']) <= 3})/3,
                team_hit=int(truth[chosen[0]]['team'] == winner['team']))


def verify_row(event, values, saved):
    selected = picks(values)
    assert selected == [saved[f'pred{i}'] for i in (1, 2, 3)]
    result = metrics(event, selected)
    for field, value in result.items():
        assert math.isclose(value, float(saved[field]), abs_tol=1e-14)
    return result


def paired(a, b):
    import f1.walkforward as W
    assert len(a) == len(b) == 61
    diff = [int(x)-int(y) for x, y in zip(a, b)]
    assert len(diff) == 61
    indices = np.random.default_rng(0).integers(0, len(diff), size=(5000, len(diff)))
    draws = sorted(math.fsum(diff[int(i)] for i in sample)/len(diff) for sample in indices)
    def quantile(p):
        rank = (len(draws)-1)*p
        lo = math.floor(rank)
        return draws[lo]+(rank-lo)*(draws[math.ceil(rank)]-draws[lo])
    independent = [math.fsum(diff)/len(diff), [quantile(.025), quantile(.975)],
                   sum(x <= 0 for x in draws)/len(draws)]
    actual = W.paired_boot(a, b, n=5000, seed=0)
    assert math.isclose(independent[0], actual[0], abs_tol=1e-14)
    np.testing.assert_allclose(independent[1], actual[1], rtol=0, atol=1e-14)
    assert independent[2] == actual[2]
    return dict(difference=independent[0], ci95=independent[1], p_nonpositive=independent[2])


def audit():
    import f1.walkforward as W
    W.assert_pre_weekend(['prior_'+m for m in MEASURES])
    paths = [ROOT/'data/results.csv', ROOT/'results/e185_inner_measurement_audit.csv',
             ROOT/'results/e181_inner_predictions.csv'] + [ROOT/'data/parts'/f for f, _ in FILES]
    before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    events = calendar(rows(paths[0]))
    assert len(events) == 103 and sum(len(e['entries']) for e in events.values()) == 2060
    targets = [key for key in sorted(events) if key[0] >= 2020]
    assert len(targets) == 61 and sum(len(events[k]['entries']) for k in targets) == 1220
    source = sources(events)
    saved = {}
    for r in rows(paths[1], 2020):
        key = (int(r['season']), int(r['round']), r['model'])
        assert key not in saved
        saved[key] = r
    assert len(saved) == 61*30
    expected_models = {prefix+'_'+m for prefix in ('PRIOR', 'NULL', 'HINDSIGHT') for m in MEASURES}
    expected_models |= {'standings', 'ens', 'win_logit'}
    assert set(saved) == {(y, r, m) for y, r in targets for m in expected_models}
    reference = {}
    for r in rows(paths[2], 2020):
        if r['model'] in ('standings', 'ens', 'win_logit'):
            key = (int(r['season']), int(r['round']), r['model'])
            group = reference.setdefault(key, {})
            assert r['driver'] not in group and math.isfinite(float(r['score']))
            group[r['driver']] = float(r['score'])
    records = {}; vectors = {}
    for model in ['PRIOR_'+m for m in MEASURES]+['standings', 'ens', 'win_logit']:
        all_metrics = []; seasons = {}
        for key in targets:
            values = (scores(events, source[model.removeprefix('PRIOR_')], key)
                      if model.startswith('PRIOR_') else reference[(*key, model)])
            assert set(values) == {r['driver'] for r in events[key]['entries']}
            measured = verify_row(events[key], values, saved[(*key, model)])
            all_metrics.append(measured)
            seasons[key[0]] = seasons.get(key[0], 0)+measured['winner']
        vectors[model] = [r['winner'] for r in all_metrics]
        records[model] = dict(races=len(all_metrics), winner_hits=sum(vectors[model]),
                              winner_rate=sum(vectors[model])/len(all_metrics),
                              strict_podium=math.fsum(r['podium'] for r in all_metrics)/len(all_metrics),
                              team_hits=sum(r['team_hit'] for r in all_metrics), season_hits=seasons)
    contrasts = {}
    for name in ('PRIOR_matched_gap', 'PRIOR_med_gap'):
        contrasts[name] = {ref: paired(vectors[name], vectors[ref])
                          for ref in ('standings', 'ens', 'win_logit', 'PRIOR_med_gap') if ref != name}
    target = (2021, 1)
    for measure in MEASURES:
        changed = {key: (-999. if events[key[0]]['k'] >= events[target]['k'] else v)
                   for key, v in source[measure].items()}
        prefix = {key: v for key, v in source[measure].items()
                  if events[key[0]]['k'] < events[target]['k']}
        for key in targets:
            if events[key]['k'] <= events[target]['k']:
                assert scores(events, source[measure], key) == scores(events, changed, key)
                assert scores(events, source[measure], key) == scores(events, prefix, key)
    earlier = dict(source['matched_gap'])
    chosen = next(key for key, v in sorted(earlier.items())
                  if events[key[0]]['k'] < events[target]['k'] and math.isfinite(v)
                  and key[1] in {r['code'] for r in events[target]['entries']})
    earlier[chosen] += 1
    assert scores(events, earlier, target) != scores(events, source['matched_gap'], target)
    poisoned = [{**r, 'order': str(len(events[(int(r['season']), int(r['round']))]['entries'])+1-int(r['order'])),
                 'grid': '-99999'} for e in events.values() for r in e['entries']]
    other_events = calendar(poisoned)
    for key in targets:
        for measure in MEASURES:
            assert scores(events, source[measure], key) == scores(other_events, source[measure], key)
    assert before == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return dict(experiment='E284', registration_commit='485de6b', models=records, paired=contrasts,
                source_hashes=before, checked_prior_top3_lists=61*len(MEASURES),
                checked_reference_top3_lists=61*3, current_future_projection_poison_and_prefix_exact=True,
                earlier_projection_response=True, actual_target_label_grid_independence=True,
                scope='Existing E185 historical results only; no new forecast fit, feature or raw-source decoder',
                limitations=['Repeatedly inspected inner cohort; all E185 admission gates remain failed.',
                             'Roster and historical source-publication assumptions remain unverified.',
                             'Not an exhaustive maximum across every historical experiment.',
                             'No outer/2026 evaluation or E278 pool expansion.'])


if __name__ == '__main__':
    result = audit()
    path = ROOT/'results/e284_inner_record_audit.json'
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
