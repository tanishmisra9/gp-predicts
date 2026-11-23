"""E278 fixed-pool hindsight coverage, NOT a deployable predictor."""
import csv
import hashlib
import json
from collections import Counter, defaultdict

import f1.walkforward as W
from f1.data import DATA, ROOT

MODELS = ['standings', 'ens', 'win_logit', 'gbm_base', 'ordered_ids']
SOURCES = [DATA / 'results.csv', ROOT / 'results/e268_inner_predictions.csv']


def read_inputs():
    truth = {}
    with SOURCES[0].open() as file:
        for row in csv.DictReader(file):
            if not 2020 <= int(row['season']) <= 2022:
                continue  # Do not interpret outer/restricted outcomes.
            key = int(row['season']), int(row['round']), row['driver']
            assert key not in truth
            truth[key] = dict(team=row['team'], order=int(row['order']))
    scores = {name: {} for name in MODELS}
    with SOURCES[1].open() as file:
        for row in csv.DictReader(file):
            assert 2020 <= int(row['season']) <= 2022
            if row['model'] not in MODELS:
                continue
            key = int(row['season']), int(row['round']), row['driver']
            model = scores[row['model']]; assert key not in model
            model[key] = float(row['score'])
    assert len(truth) == 1220 and len({k[:2] for k in truth}) == 61
    return truth, scores


def analyze(truth, scores):
    import math
    assert truth and set(scores) == set(MODELS)
    for name in MODELS:
        assert set(scores[name]) == set(truth) and all(math.isfinite(s) for s in scores[name].values())
    events = defaultdict(list)
    for (year, rnd, driver), record in truth.items():
        assert 2020 <= year <= 2022 and driver and record['team']
        events[year, rnd].append(driver)
    rows = []; hit_vectors = {name: [] for name in MODELS}; overlaps = {name: [] for name in MODELS}; teams = Counter()
    for (year, rnd), drivers in sorted(events.items()):
        key = lambda d: (year, rnd, d)
        actual = sorted(drivers, key=lambda d: truth[key(d)]['order'])
        assert len(actual) >= 3 and sorted(truth[key(d)]['order'] for d in actual) == list(range(1, len(actual)+1))
        winner = actual[0]; winning_team = truth[key(winner)]['team']
        picks = {}; picked_teams = set()
        for name in MODELS:
            pred = sorted(drivers, key=lambda d: (scores[name][key(d)], d)); picks[name] = pred[0]
            hit_vectors[name].append(int(pred[0] == winner)); overlaps[name].append(len(set(pred[:3]) & set(actual[:3]))/3)
            team = truth[key(pred[0])]['team']; picked_teams.add(team); teams[name] += int(team == winning_team)
        chosen = set(picks.values()); oracle = int(winner in chosen)
        partition = ('unanimous_correct' if oracle else 'unanimous_wrong') if len(chosen) == 1 else ('disagreement_any_correct' if oracle else 'disagreement_none_correct')
        miss = None if oracle else 'winning_team_covered_wrong_driver' if winning_team in picked_teams else 'winning_team_absent'
        rows.append(dict(season=year, round=rnd, winner=winner, winning_team=winning_team, picks=picks,
                         picked_teams=sorted(picked_teams), distinct_drivers=len(chosen), distinct_teams=len(picked_teams),
                         oracle_hit=oracle, partition=partition, miss_type=miss))
    n = len(rows); oracle = [r['oracle_hit'] for r in rows]
    models = {name: dict(winners=sum(hits), winner_rate=sum(hits)/n, team_hits=teams[name], strict_podium_overlap=sum(overlaps[name])/n) for name, hits in hit_vectors.items()}
    paired = {}
    for name in ['standings', 'ens', 'win_logit']:
        diff, ci, p = W.paired_boot(oracle, hit_vectors[name], n=5000, seed=0)
        paired[name] = dict(diff=float(diff), ci=ci.tolist(), p_nonpositive=float(p))
    summary = dict(races=n, entrants=len(truth), models=models, oracle_winners=sum(oracle), oracle_rate=sum(oracle)/n,
                   numerical_75_count=(3*n+3)//4, pool_reaches_numerical_75=bool(4*sum(oracle) >= 3*n),
                   partitions=dict(Counter(r['partition'] for r in rows)), misses=dict(Counter(r['miss_type'] for r in rows if r['miss_type'])),
                   by_season={str(year): dict(races=sum(r['season'] == year for r in rows), oracle_winners=sum(r['oracle_hit'] for r in rows if r['season'] == year),
                                             partitions=dict(Counter(r['partition'] for r in rows if r['season'] == year)),
                                             misses=dict(Counter(r['miss_type'] for r in rows if r['season'] == year and r['miss_type']))) for year in sorted({r['season'] for r in rows})},
                   hindsight_paired_headroom=paired, diagnostic_only=True, trained_selector=False, outer_evaluated=False, evaluated_2026=False,
                   universal_accuracy_ceiling=False)
    return rows, summary


def main():
    output = ROOT / 'results/e278_winner_pool_summary.json'
    assert not output.exists(), 'do not overwrite completed diagnostic'
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}
    truth, scores = read_inputs(); rows, summary = analyze(truth, scores)
    assert summary['races'] == 61 and summary['entrants'] == 1220
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}
    path = ROOT / 'results/e278_winner_pool_events.json'; path.write_text(json.dumps(rows, indent=2)+'\n')
    summary.update(pool=MODELS, source_hashes=before, source_immutable=True,
                   code_sha256=hashlib.sha256((ROOT / 'experiments/e278_winner_pool.py').read_bytes()).hexdigest(),
                   events_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    output.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
