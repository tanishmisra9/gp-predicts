"""E183 history continuity must not import target/future observations."""
import numpy as np
import pandas as pd

from experiments.e183_lineage_winner import lineage_features
from f1.features import build_features


def test_continuity():
    df = pd.DataFrame(dict(season=[2019,2019,2020,2020], round=[1,1,1,1],
        driver=['a','b','a','b'], code=['A','B','A','B'], team=['renault','renault','alpine','alpine'],
        circuit=['track']*4, grid=[1,2,1,2], order=[1,2,1,2], finish=[1,2,1,2], status=['Finished']*4,
        points=[25.,18.,25.,18.], laps=[50]*4, q_pos=[1.,2.,1.,2.], q1=[80.,81.,80.,81.],
        q2=[np.nan]*4, q3=[np.nan]*4, s_grid=[np.nan]*4, s_order=[np.nan]*4))
    pace = pd.DataFrame(dict(season=df.season, round=df['round'], code=df.code, pace_gap=[0.,1.,0.,1.]))
    x, base, candidate = lineage_features(df, pace)
    raw = build_features(df, pace)
    pd.testing.assert_frame_equal(x[raw.columns], raw, check_exact=True)
    assert len(base) == len(candidate) and len(candidate) == len(set(candidate))
    current = x.season.eq(2020)
    assert x.loc[current, 't3_order'].isna().all()
    np.testing.assert_allclose(x.loc[current, 'lineage_t3_order'], 1.5)
    assert x.loc[x.season.eq(2019), 'lineage_t3_order'].isna().all()
    assert x.loc[current, 'team'].eq('alpine').all()
    for invalid in (pd.concat([df, df.iloc[:1]]), df.assign(season=2023)):
        try:
            lineage_features(invalid, pace)
        except AssertionError:
            pass
        else:
            raise AssertionError('invalid source accepted')
    unknown = df.assign(team='unknown_constructor')
    u, _, _ = lineage_features(unknown, pace)
    for c in base:
        if 'lineage_' + c in u:
            pd.testing.assert_series_equal(u[c], u['lineage_' + c], check_names=False, check_exact=True)
    try:
        lineage_features(pd.concat([df, df.iloc[:1]]), pace)
    except AssertionError:
        pass
    else:
        raise AssertionError('duplicate entrant accepted')


def test_source_cutoff():
    df = pd.read_csv('data/results.csv').loc[lambda d: d.season.le(2021)].copy()
    pace = pd.read_csv('data/racepace_ff1.csv').loc[lambda d: d.season.le(2021)].copy()
    before, _, F = lineage_features(df, pace)
    target = before.season.eq(2021) & before['round'].eq(1)
    changed = df.copy(); mask = changed.season.ge(2021)
    rng = np.random.default_rng(183)
    for c in ['grid','order','finish','points','laps','q_pos','q1','q2','q3','s_grid','s_order']:
        changed.loc[mask, c] = rng.integers(1,100,mask.sum())
    changed.loc[mask, 'status'] = 'Collision'
    badpace = pace.copy(); badpace.loc[badpace.season.ge(2021), 'pace_gap'] = 999.
    after, _, _ = lineage_features(changed, badpace)
    pd.testing.assert_frame_equal(before.loc[target,F], after.loc[target,F], check_exact=True)
    # Changing earlier Renault observations must reach Alpine's inherited features.
    changed = df.copy(); prior = changed.season.eq(2020) & changed.team.eq('renault')
    changed.loc[prior, ['order','q1','q2','q3']] = 90.
    earlier, _, _ = lineage_features(changed, pace)
    alpine = target & before.team.eq('alpine')
    assert not before.loc[alpine,'lineage_t3_order'].equals(earlier.loc[alpine,'lineage_t3_order'])
    # Pace has its own source and cutoff, independently of official results.
    paceonly, _, _ = lineage_features(df, badpace)
    pd.testing.assert_frame_equal(before.loc[target,F], paceonly.loc[target,F], check_exact=True)
    olderpace = pace.copy(); olderpace.loc[olderpace.season.eq(2020), 'pace_gap'] = 777.
    response, _, _ = lineage_features(df, olderpace)
    assert not before.loc[target,'lineage_t3_pace_gap'].equals(response.loc[target,'lineage_t3_pace_gap'])


if __name__ == '__main__':
    test_continuity()
    test_source_cutoff()
    print('lineage continuity, unchanged raw features, identities, cutoff and source responsiveness passed')
