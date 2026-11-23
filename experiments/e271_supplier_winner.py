"""E271 conditional retrospective supplier-peer winner test; no new acquisition."""
import hashlib
import json

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.e168_winner import win_score
from experiments.e270_supplier_records import LEGACY
from f1.data import DATA, ROOT
from f1.features import build_features, pre_weekend_cols
from f1.profile import PU

KEYS = ['season', 'round', 'code']
NEW = ['supplier_peer_points', 'legacy_peer_points']
CONTROLS = ['legacy_win', 'win_logit', 'standings', 'ens']


def load_inputs():
    audit = json.loads((ROOT / 'results/e270_independent_audit.json').read_text())
    assert audit['recorded_brand_source_gate']
    for name in ['e270_projection.json', 'e270_identity_audit.json', 'e270_summary.json']:
        assert hashlib.sha256((ROOT / 'results' / name).read_bytes()).hexdigest() == audit['artifact_sha256'][name]
    results = pd.read_csv(DATA / 'results.csv', parse_dates=['date']).loc[lambda d: d.season.between(2018, 2022)]
    pace = pd.read_csv(DATA / 'racepace_ff1.csv').loc[lambda d: d.season.between(2018, 2022)]
    projections = json.loads((ROOT / 'results/e270_projection.json').read_text())[:103]
    identities = json.loads((ROOT / 'results/e270_identity_audit.json').read_text())
    return results, pace, projections, identities


def source_rows(results, projections, identities):
    assert not results.duplicated(KEYS).any() and results.season.between(2018, 2022).all()
    official = set(map(tuple, results[KEYS].to_numpy()))
    events = {(s, r) for s, r, _ in official}
    maps = {(r['season'], r['round']): r['identities'] for r in identities}
    assert len(maps) == len(identities) and set(maps) <= events
    seen = set(); rows = []
    for event in projections:
        key = event['season'], event['round']
        assert key in events and key not in seen and key in maps
        seen.add(key); identity = maps[key]
        assert len(set(identity.values())) == len(identity)
        numbers = [r['driverNumber'] for r in event['rows'] if r['driverNumber'] is not None]
        assert len(numbers) == len(set(numbers))
        constructor_brands = {}
        for row in event['rows']:
            constructor, brand = row['constructorId'], row['engineManufacturerId']
            if constructor and brand:
                assert constructor_brands.setdefault(constructor, brand) == brand
            code = identity.get(row['driverNumber'])
            if code is None:
                continue  # E270's unmatched Q identities remain unknown.
            assert (*key, code) in official
            covered = all(row.get(f) for f in ['driverId', 'constructorId', 'engineManufacturerId'])
            rows.append(dict(season=key[0], round=key[1], code=code, brand=brand if covered else None))
    source = pd.DataFrame(rows, columns=KEYS+['brand'])
    assert not source.duplicated(KEYS).any()
    return source


def add_history(X, results, source):
    assert X.index.equals(pd.RangeIndex(len(X)))
    for frame in [X, results, source]:
        assert not frame.duplicated(KEYS).any() and frame.season.between(2018, 2022).all()
    assert np.isfinite(results.points.to_numpy(float)).all() and results.points.ge(0).all()
    metadata = X[KEYS+['k', 'team']].merge(results[KEYS+['date', 'points', 'team']], on=KEYS,
                                        suffixes=('', '_official'), validate='one_to_one')
    assert len(metadata) == len(X) == len(results) and metadata.team.equals(metadata.team_official)
    assert metadata.date.notna().all() and X.date.equals(metadata.date)
    assert source.brand.map(lambda b: pd.isna(b) or isinstance(b, str) and bool(b.strip())).all()
    assert len(metadata.merge(source[KEYS], on=KEYS, validate='one_to_one')) == len(source)
    metadata['date'] = pd.to_datetime(metadata.date, utc=True)
    events = metadata[['season', 'round', 'k', 'date']].drop_duplicates()
    assert not events.duplicated(['season', 'round']).any() and not events.k.duplicated().any()
    assert events.sort_values('k').date.is_monotonic_increasing
    history = metadata.merge(source, on=KEYS, how='left', validate='one_to_one')
    history['legacy_brand'] = [next((LEGACY.get(mapping.get(team)) for (a, b), mapping in PU.items()
                                   if a <= year <= b), None) for year, team in zip(history.season, history.team)]
    field_mean = (history.points/25).groupby(history.k).mean()
    team_rows = []
    for kind, col in [('supplier', 'brand'), ('legacy', 'legacy_brand')]:
        for (year, rnd, team), group in history[history[col].notna()].groupby(['season', 'round', 'team'], sort=True):
            assert group[col].nunique() == 1, 'contradictory team brands'
            k = int(group.k.iloc[0])
            team_rows.append(dict(kind=kind, season=int(year), round=int(rnd), team=team, k=k,
                                  date=group.date.iloc[0], brand=group[col].iloc[0],
                                  z=float((group.points/25).mean()-field_mean.loc[k])))
    teams = pd.DataFrame(team_rows, columns=['kind', 'season', 'round', 'team', 'k', 'date', 'brand', 'z'])
    out = X.copy()
    for kind, feature in zip(['supplier', 'legacy'], NEW):
        out[feature] = 0.; out[kind+'_brand'] = None
        for event in events.sort_values('k').itertuples(index=False):
            date = pd.Timestamp(event.date).normalize()
            cutoff = date-pd.Timedelta(days=(date.weekday()-3) % 7)
            past = teams[teams.kind.eq(kind) & teams.season.eq(event.season) & teams.k.lt(event.k)
                         & teams.date.lt(cutoff)]
            current = out[out.k.eq(event.k)]
            for team, group in current.groupby('team', sort=True):
                own = past[past.team.eq(team)].sort_values('k')
                if own.empty:
                    continue
                brand = own.brand.iloc[-1]
                peers = past[past.brand.eq(brand) & ~past.team.eq(team)]
                weights = .5**((event.k-peers.k)/10)
                out.loc[group.index, feature] = float((weights*peers.z).sum()/(5+weights.sum()))
                out.loc[group.index, kind+'_brand'] = brand
    return out, teams


def make_features(results, pace, source):
    assert pace.season.between(2018, 2022).all()
    X = build_features(results, pace).merge(results[KEYS+['date']], on=KEYS, validate='one_to_one')
    base = pre_weekend_cols(X)
    X, teams = add_history(X, results, source)
    W.assert_pre_weekend(base+NEW)
    return X, base, teams


def score(train, current, features, shuffled=False):
    W.assert_pre_weekend(features)
    assert len(train) and len(current) and train.k.max() < current.k.min()
    assert current.k.nunique() == 1 and current.date.nunique() == 1
    date = pd.Timestamp(current.date.iloc[0]).normalize()
    cutoff = date-pd.Timedelta(days=(date.weekday()-3) % 7)
    assert train.date.lt(cutoff).all()
    assert not train.duplicated(['k', 'driver']).any() and not current.driver.duplicated().any()
    assert train.groupby('k').win.sum().eq(1).all()
    return win_score(train, current, features, shuffled=shuffled, seed=271)


def main():
    assert not (ROOT / 'results/e271_inner_predictions.csv').exists(), 'do not overwrite a scored run'
    results, pace, projections, identities = load_inputs()
    source = source_rows(results, projections, identities)
    X, base, teams = make_features(results, pace, source)
    candidate = base+[NEW[0]]; legacy = base+[NEW[1]]
    for _, current in X[X.season.between(2020, 2022)].groupby('k'):
        training = X[X.k.lt(current.k.iloc[0]) & X.season.ge(2019)]
        assert len(training) and training.date.lt(current.date.iloc[0]-pd.Timedelta(days=3)).all()
    W.MODELS.update(
        standings=lambda tr, te, F: (-te.d_season_pts.fillna(0)*100+te.d10_order.fillna(11)*1e-3).to_numpy(),
        pre_logit=lambda tr, te, F: W.m_logit_pod(tr, te, base, C=.03, hl=100),
        win_logit=lambda tr, te, F: score(tr, te, base),
        supplier_win=lambda tr, te, F: score(tr, te, candidate),
        legacy_win=lambda tr, te, F: score(tr, te, legacy),
        supplier_null=lambda tr, te, F: score(tr, te, candidate, shuffled=True))
    names = ['standings', 'pre_logit', 'win_logit', 'supplier_win', 'legacy_win', 'supplier_null']
    R, P = W.walk_forward(X, [2020, 2021, 2022], names, F=base+NEW,
                          ens={'ens': ['standings', 'pre_logit']}, tie=None)
    assert len(R) == 61*7 and len(P) == 1220*7
    old = pd.read_csv(ROOT / 'results/e268_inner_predictions.csv', float_precision='round_trip')
    for name in ['standings', 'ens', 'win_logit', 'pre_logit']:
        check = P[P.model.eq(name)].merge(old[old.model.eq(name)][['season', 'round', 'driver', 'score']],
                                        on=['season', 'round', 'driver'], suffixes=('_new', '_old'), validate='one_to_one')
        assert len(check) == 1220
        np.testing.assert_allclose(check.score_new, check.score_old, atol=1e-14, rtol=0)
    truth = X[['season', 'round', 'driver', 'team', 'order', 'win']]
    P = P.merge(truth, on=['season', 'round', 'driver'], validate='many_to_one')
    probability_rows = []
    for (year, rnd, name), group in P[~P.model.isin(['standings', 'ens'])].groupby(['season', 'round', 'model']):
        raw = -group.score.to_numpy(); assert (raw > 0).all() and (raw < 1).all()
        probability = raw/raw.sum(); y = group.win.to_numpy()
        probability_rows.append(dict(season=int(year), round=int(rnd), model=name,
                                     logloss=float(-np.log(probability[y == 1][0])),
                                     brier=float(np.sum((probability-y)**2))))
    C = pd.DataFrame(probability_rows)
    winners = truth[truth.win.eq(1)].set_index(['season', 'round']).team
    picks = P.sort_values(['score', 'driver']).drop_duplicates(['season', 'round', 'model'])
    models = {}
    for name, rows in R.groupby('model'):
        chosen = picks[picks.model.eq(name)]
        models[name] = dict(winners=int(rows.winner.sum()), winner_rate=float(rows.winner.mean()),
                            overlap=float(rows.podium.mean()), by_season=rows.groupby('season').winner.sum().astype(int).to_dict(),
                            team_hits=sum(t == winners.loc[(s, r)] for s, r, t in chosen[['season', 'round', 'team']].itertuples(index=False, name=None)))
        if name not in ['standings', 'ens']:
            models[name].update(C[C.model.eq(name)][['logloss', 'brier']].mean().to_dict())
    wide = R.pivot(index=['season', 'round'], columns='model', values='winner').sort_index()
    paired = {}
    for name in CONTROLS:
        diff, ci, p = W.paired_boot(wide.supplier_win, wide[name], n=5000, seed=0)
        paired[name] = dict(diff=float(diff), ci=ci.tolist(), p_nonpositive=float(p))
    unknown = X[X.season.ge(2020)].assign(supplier_unknown=lambda d: d.supplier_brand.isna(), legacy_unknown=lambda d: d.legacy_brand.isna())
    summary = dict(models=models, paired=paired, candidate='supplier_win',
                   gate=bool(models['supplier_win']['winners'] >= 37 and all(v['ci'][0] > 0 for v in paired.values())
                             and models['supplier_win']['winners'] > models['supplier_null']['winners']),
                   base_features=base, candidate_features=candidate, legacy_features=legacy,
                   winner_changes={name: int(picks[picks.model.eq('supplier_win')].set_index(['season', 'round']).driver.ne(
                       picks[picks.model.eq(name)].set_index(['season', 'round']).driver).sum()) for name in CONTROLS},
                   unknown_by_event=unknown.groupby(['season', 'round'])[['supplier_unknown', 'legacy_unknown']].mean().reset_index().to_dict('records'),
                   unknown_by_season=unknown.groupby('season')[['supplier_unknown', 'legacy_unknown']].mean().reset_index().to_dict('records'),
                   historical_publication_time_verified=False, physical_oem_verified=False,
                   outer_evaluated=False, evaluated_2026=False, evaluated_races=61, forecast_entrants=1220,
                   input_hashes={name: hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
                                 for name, frame in [('results_through_2022', results), ('pace_through_2022', pace), ('supplier_source', source)]})
    X[KEYS+['k']+NEW+['supplier_brand', 'legacy_brand']].to_csv(ROOT / 'results/e271_features.csv', index=False)
    teams.to_csv(ROOT / 'results/e271_source_teams.csv', index=False)
    for suffix, frame in [('per_race', R), ('predictions', P), ('probability_metrics', C)]:
        frame.to_csv(ROOT / f'results/e271_inner_{suffix}.csv', index=False)
    summary['artifact_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'results').glob('e271_*.csv')}
    (ROOT / 'results/e271_inner_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('RESULT', json.dumps(dict(models=models, paired=paired, gate=summary['gate'])), flush=True)


if __name__ == '__main__':
    main()
