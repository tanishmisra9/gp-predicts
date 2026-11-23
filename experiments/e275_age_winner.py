"""E275 fixed chronological age/experience winner ablation, inner years only."""
import hashlib
import json

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.e168_winner import win_score
from experiments.e274_birth_source import birth_date, prior_records
from f1.data import DATA, ROOT
from f1.features import build_features, pre_weekend_cols

KEYS = ['season', 'round', 'driver']
EXPERIENCE = ['observed_entry_count', 'observed_rookie_flag']
AGES = ['prior_age_years', 'prior_age_squared']
NEW = EXPERIENCE+AGES
CONTROLS = ['experience_win', 'win_logit', 'standings', 'ens']


def cutoff_date(date):
    date = pd.Timestamp(date).normalize()
    return date-pd.Timedelta(days=(date.weekday()-3) % 7)


def load_inputs():
    audit = json.loads((ROOT / 'results/e274_independent_audit.json').read_text())
    assert audit['source_gate']
    for name in ['e274_birth_projection.json', 'e274_birth_source_summary.json']:
        assert hashlib.sha256((ROOT / 'results' / name).read_bytes()).hexdigest() == audit['output_hashes'][name]
    results = pd.read_csv(DATA / 'results.csv', parse_dates=['date']).loc[lambda d: d.season.between(2018, 2022)]
    pace = pd.read_csv(DATA / 'racepace_ff1.csv').loc[lambda d: d.season.between(2018, 2022)]
    source = json.loads((ROOT / 'results/e274_birth_projection.json').read_text())
    return results, pace, source


def add_history(X, source):
    assert X.index.equals(pd.RangeIndex(len(X))) and not X.duplicated(KEYS).any()
    assert X.season.between(2018, 2022).all() and X.date.notna().all()
    events = X[['season', 'round', 'k', 'date']].drop_duplicates()
    assert not events.duplicated(['season', 'round']).any() and not events.k.duplicated().any()
    assert events.sort_values('k').date.is_monotonic_increasing
    calendar = {(r.season, r.round, r.driver): r.date.strftime('%Y-%m-%d') for r in X.itertuples()}
    source_keys = [(r['season'], r['round'], r['driver']) for r in source]
    assert len(source_keys) == len(set(source_keys)) and set(source_keys) <= set(calendar)
    clean_source = []
    for row, key in zip(source, source_keys):
        assert row['date'] == calendar[key]
        clean_source.append(dict(row, dateOfBirth=birth_date(row['dateOfBirth'], row['date'])))
    targets = [dict(season=r.season, round=r.round, driver=r.driver, date=r.date.strftime('%Y-%m-%d')) for r in X.itertuples()]
    prior = prior_records(clean_source, targets)
    out = X.copy(); out['prior_dob'] = [r['prior_dob'] for r in prior]
    out[EXPERIENCE[0]] = 0
    for event in events.itertuples(index=False):
        past = X[X.k.lt(event.k) & X.date.lt(cutoff_date(event.date))]
        counts = past.groupby('driver').size()
        current = X.k.eq(event.k)
        out.loc[current, EXPERIENCE[0]] = X.loc[current, 'driver'].map(counts).fillna(0).astype(int)
    out[EXPERIENCE[1]] = out[EXPERIENCE[0]].lt(10).astype(float)
    out[AGES[0]] = [(cutoff_date(date)-pd.Timestamp(dob)).days/365.2425 if dob is not None else np.nan
                    for date, dob in zip(out.date, out.prior_dob)]
    out[AGES[1]] = out[AGES[0]]**2
    return out


def make_features(results, pace, source):
    assert results.season.between(2018, 2022).all() and pace.season.between(2018, 2022).all()
    assert not results.duplicated(KEYS).any()
    X = build_features(results, pace).merge(results[KEYS+['date']], on=KEYS, validate='one_to_one')
    base = pre_weekend_cols(X)
    X = add_history(X, source)
    W.assert_pre_weekend(base+NEW)
    return X, base


def score(train, current, features, shuffled=False):
    W.assert_pre_weekend(features)
    assert len(train) and len(current) and current.k.nunique() == 1 and current.date.nunique() == 1
    assert train.k.max() < current.k.min() and train.date.lt(cutoff_date(current.date.iloc[0])).all()
    assert not train.duplicated(['k', 'driver']).any() and not current.driver.duplicated().any()
    assert train.groupby('k').win.sum().eq(1).all()
    return win_score(train, current, features, shuffled=shuffled, seed=275)


def main():
    assert not (ROOT / 'results/e275_inner_predictions.csv').exists(), 'do not overwrite a scored experiment'
    results, pace, source = load_inputs()
    X, base = make_features(results, pace, source)
    candidate = base+NEW; experience = base+EXPERIENCE
    for _, current in X[X.season.between(2020, 2022)].groupby('k'):
        train = X[X.k.lt(current.k.iloc[0]) & X.season.ge(2019)]
        assert len(train) and train.date.lt(cutoff_date(current.date.iloc[0])).all()
    W.MODELS.update(
        standings=lambda tr, te, F: (-te.d_season_pts.fillna(0)*100+te.d10_order.fillna(11)*1e-3).to_numpy(),
        pre_logit=lambda tr, te, F: W.m_logit_pod(tr, te, base, C=.03, hl=100),
        win_logit=lambda tr, te, F: score(tr, te, base),
        age_win=lambda tr, te, F: score(tr, te, candidate),
        experience_win=lambda tr, te, F: score(tr, te, experience),
        age_null=lambda tr, te, F: score(tr, te, candidate, shuffled=True))
    R, P = W.walk_forward(X, [2020, 2021, 2022], ['standings', 'pre_logit', 'win_logit', 'age_win', 'experience_win', 'age_null'],
                          F=candidate, ens={'ens': ['standings', 'pre_logit']}, tie=None)
    assert len(R) == 61*7 and len(P) == 1220*7
    old = pd.read_csv(ROOT / 'results/e271_inner_predictions.csv', float_precision='round_trip')
    for name in ['standings', 'ens', 'win_logit', 'pre_logit']:
        joined = P[P.model.eq(name)].merge(old[old.model.eq(name)][KEYS+['score']], on=KEYS,
                                         suffixes=('_new', '_old'), validate='one_to_one')
        assert len(joined) == 1220
        np.testing.assert_allclose(joined.score_new, joined.score_old, atol=1e-14, rtol=0)
    truth = X[KEYS+['team', 'order', 'win']]
    P = P.merge(truth, on=KEYS, validate='many_to_one')
    probability_rows = []
    for (year, rnd, name), group in P[~P.model.isin(['standings', 'ens'])].groupby(['season', 'round', 'model']):
        raw = -group.score.to_numpy(); assert (raw > 0).all() and (raw < 1).all()
        p = raw/raw.sum(); y = group.win.to_numpy()
        probability_rows.append(dict(season=int(year), round=int(rnd), model=name,
                                     logloss=float(-np.log(p[y == 1][0])), brier=float(np.sum((p-y)**2))))
    C = pd.DataFrame(probability_rows)
    winners = truth[truth.win.eq(1)].set_index(['season', 'round']).team
    picks = P.sort_values(['score', 'driver']).drop_duplicates(['season', 'round', 'model'])
    models = {}
    for name, rows in R.groupby('model'):
        chosen = picks[picks.model.eq(name)]
        models[name] = dict(winners=int(rows.winner.sum()), winner_rate=float(rows.winner.mean()), overlap=float(rows.podium.mean()),
                            by_season=rows.groupby('season').winner.sum().astype(int).to_dict(),
                            team_hits=sum(t == winners.loc[(s, r)] for s, r, t in chosen[['season', 'round', 'team']].itertuples(index=False, name=None)))
        if name not in ['standings', 'ens']:
            models[name].update(C[C.model.eq(name)][['logloss', 'brier']].mean().to_dict())
    wide = R.pivot(index=['season', 'round'], columns='model', values='winner').sort_index()
    paired = {}
    for name in CONTROLS:
        diff, ci, p = W.paired_boot(wide.age_win, wide[name], n=5000, seed=0)
        paired[name] = dict(diff=float(diff), ci=ci.tolist(), p_nonpositive=float(p))
    unknown = X[X.season.ge(2020)].assign(dob_unknown=lambda d: d.prior_dob.isna())
    summary = dict(models=models, paired=paired, candidate='age_win',
                   gate=bool(models['age_win']['winners'] >= 37 and all(v['ci'][0] > 0 for v in paired.values())
                             and models['age_win']['winners'] > models['age_null']['winners']),
                   base_features=base, candidate_features=candidate, experience_features=experience,
                   winner_changes={name: int(picks[picks.model.eq('age_win')].set_index(['season', 'round']).driver.ne(
                       picks[picks.model.eq(name)].set_index(['season', 'round']).driver).sum()) for name in CONTROLS},
                   unknown_by_event=unknown.groupby(['season', 'round']).dob_unknown.mean().reset_index().to_dict('records'),
                   unknown_by_season=unknown.groupby('season').dob_unknown.mean().reset_index().to_dict('records'),
                   historical_publication_time_verified=False, outer_evaluated=False, evaluated_2026=False,
                   evaluated_races=61, forecast_entrants=1220,
                   model_config=dict(C=.03, max_iter=500, training_half_life=100, null_seed=275),
                   code_hashes={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                                ['experiments/e275_age_winner.py', 'experiments/e274_birth_source.py',
                                 'experiments/e168_winner.py', 'f1/features.py', 'f1/walkforward.py']},
                   input_hashes={name: hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
                                 for name, frame in [('results_through_2022', results), ('pace_through_2022', pace), ('birth_source', pd.DataFrame(source))]})
    X[KEYS+['k', 'prior_dob']+NEW].to_csv(ROOT / 'results/e275_features.csv', index=False)
    for suffix, frame in [('per_race', R), ('predictions', P), ('probability_metrics', C)]:
        frame.to_csv(ROOT / f'results/e275_inner_{suffix}.csv', index=False)
    summary['artifact_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'results').glob('e275_*.csv')}
    (ROOT / 'results/e275_inner_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('RESULT', json.dumps(dict(models=models, paired=paired, gate=summary['gate'])), flush=True)


if __name__ == '__main__':
    main()
