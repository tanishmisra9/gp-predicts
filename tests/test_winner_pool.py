"""Synthetic fixed-pool diagnostic contracts; no real winner inspection."""
import copy

from experiments.e278_winner_pool import MODELS, analyze


def main():
    truth = {(2020, r, d): dict(team=t, order=i+1) for r in range(1, 5) for i, (d, t) in enumerate([('a', 'x'), ('b', 'x'), ('c', 'y')])}
    scores = {m: {k: float({'a': 0, 'b': 1, 'c': 2}[k[2]]) for k in truth} for m in MODELS}
    # R1 unanimous correct; R2 disagreement with a winner; R3 wrong teammate;
    # R4 unanimous wrong team.
    scores[MODELS[0]][2020, 2, 'c'] = -1
    for model in MODELS:
        scores[model][2020, 3, 'b'] = -1
        scores[model][2020, 4, 'c'] = -1
    rows, s = analyze(truth, scores)
    assert s['oracle_winners'] == 2 and s['numerical_75_count'] == 3 and not s['pool_reaches_numerical_75']
    assert s['partitions'] == dict(unanimous_correct=1, disagreement_any_correct=1, unanimous_wrong=2)
    assert s['misses'] == dict(winning_team_covered_wrong_driver=1, winning_team_absent=1)
    changed = copy.deepcopy(scores); changed[MODELS[0]][2020, 4, 'a'] = -2
    _, c = analyze(truth, changed)
    assert c['oracle_winners'] == 3 and c['pool_reaches_numerical_75']
    tie = copy.deepcopy(scores)
    for model in MODELS: tie[model][2020, 4, 'a'] = -1
    assert analyze(truth, tie)[0][-1]['oracle_hit'] == 1
    malformed = []
    missing = copy.deepcopy(scores); missing[MODELS[0]].pop((2020, 1, 'a')); malformed.append((truth, missing))
    bad = copy.deepcopy(scores); bad[MODELS[0]][2020, 1, 'a'] = float('nan'); malformed.append((truth, bad))
    extra = copy.deepcopy(truth); extra[2026, 1, 'a'] = dict(team='x', order=1); malformed.append((extra, scores))
    duplicate = copy.deepcopy(truth); duplicate[2020, 1, 'b']['order'] = 1; malformed.append((duplicate, scores))
    for t, p in malformed:
        try: analyze(t, p)
        except AssertionError: pass
        else: raise AssertionError('invalid cohort accepted')
    print('E278 synthetic coverage/partition/75-count/tie/positive score response/missing/nonfinite/extra-year/duplicate-order checks PASS')


if __name__ == '__main__':
    main()
