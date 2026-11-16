"""E185 prior measurements are strictly causal; hindsight stays diagnostic."""
import numpy as np
import pandas as pd

from experiments.e185_pace_winner_audit import MEASURES, prior_measurements, shuffled_source, source_table


def test_decay_and_missing():
    entries = pd.DataFrame(dict(season=[2020]*6,round=[1,1,2,2,3,3],k=[0,0,1,1,2,2],driver=['a','b']*3,code=['A','B']*3))
    source = entries[['season','round','code']].copy()
    for col in MEASURES:
        source[col] = [0.,np.nan,2.,np.nan,99.,99.]
    got = prior_measurements(entries,source)
    assert got.loc[entries.k.eq(0),MEASURES].isna().all().all()
    assert got.loc[entries.code.eq('B'),MEASURES].isna().all().all()
    w0,w1 = .5**(.2),.5**(.1)
    np.testing.assert_allclose(got.loc[entries.k.eq(2)&entries.code.eq('A'),MEASURES],2*w1/(w0+w1))
    changed=source.copy(); changed.loc[changed['round'].ge(3),MEASURES]=-999.
    after=prior_measurements(entries,changed)
    pd.testing.assert_frame_equal(got,after,check_exact=True)
    changed.loc[changed['round'].eq(1)&changed.code.eq('A'),MEASURES]=10.
    after=prior_measurements(entries,changed)
    assert not got.loc[entries.k.eq(2),MEASURES].equals(after.loc[entries.k.eq(2),MEASURES])
    shuffled=shuffled_source(source)
    pd.testing.assert_frame_equal(shuffled,shuffled_source(source),check_exact=True)
    for key,g in source.groupby(['season','round']):
        h=shuffled[shuffled.season.eq(key[0])&shuffled['round'].eq(key[1])]
        original=sorted(map(tuple,g[MEASURES].fillna(-1e9).to_numpy()))
        assigned=sorted(map(tuple,h[MEASURES].fillna(-1e9).to_numpy()))
        assert original==assigned


def test_real_source():
    df=pd.read_csv('data/results.csv').loc[lambda d:d.season.le(2022)]
    entries=df[['season','round','driver','code']].copy()
    entries['k']=entries.groupby(['season','round'],sort=True).ngroup()
    source=source_table()
    before=prior_measurements(entries,source)
    target=entries.season.eq(2021)&entries['round'].eq(1)
    changed=source.copy()
    future=changed.season.ge(2021)
    changed.loc[future,MEASURES]=999.
    after=prior_measurements(entries,changed)
    pd.testing.assert_frame_equal(before.loc[target,MEASURES],after.loc[target,MEASURES],check_exact=True)
    changed.loc[changed.season.eq(2020),MEASURES]=-999.
    after=prior_measurements(entries,changed)
    assert not before.loc[target,MEASURES].equals(after.loc[target,MEASURES])
    assert source.season.max()==2022


if __name__=='__main__':
    test_decay_and_missing()
    test_real_source()
    print('pace audit decay, missing history, null integrity and real-source cutoff checks passed')
