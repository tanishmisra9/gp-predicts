"""E212 source aggregation and calendar-prefix history across seasons."""
import numpy as np
import pandas as pd

from experiments.e212_braking_winner import NEW,add_history,allowed_urls,observations


def fixture():
    rows=[]; laps=[]
    for k in range(8):
        season=2018+k//4; round_=k%4+1
        for team in range(6):
            for code in (f'a{team}',f'b{team}'):
                rows.append(dict(season=season,round=round_,k=k,code=code,team=f't{team}'))
                for lap in range(2,12):
                    laps.append(dict(season=season,round=round_,driver=code,lap=lap,duration=1+team*.05))
    return pd.DataFrame(rows),pd.DataFrame(laps)


def test_history():
    X,source=fixture(); entries=X[['season','round','code','team']]
    table=observations(source,entries); output=add_history(X,table)
    assert table.centered_response.notna().all() and output[NEW].notna().all().all()
    expected=-.125*.5**.1/(5+.5**.1)
    np.testing.assert_allclose(output.loc[X.k.eq(1)&X.team.eq('t0'),NEW],expected)
    changed=table.copy(); changed.loc[changed.season.eq(2019),'centered_response']=9
    again=add_history(X,changed)
    np.testing.assert_array_equal(output.loc[X.k.le(4),NEW],again.loc[X.k.le(4),NEW])
    prior=table.copy(); prior.loc[prior.season.eq(2018)&prior['round'].eq(1),'centered_response']+=1
    assert not add_history(X,prior).loc[X.k.eq(4),NEW].equals(output.loc[X.k.eq(4),NEW])
    prefix=add_history(X,table[table.season.eq(2018)])
    np.testing.assert_array_equal(output.loc[X.k.eq(4),NEW],prefix.loc[X.k.eq(4),NEW])
    assert add_history(X,table.iloc[:0])[NEW].eq(0).all().all()
    assert observations(source[source.lap.ne(2)],entries).response.isna().all()
    for bad in (pd.concat([source,source.iloc[:1]]),source.assign(season=2026),source.assign(driver='unknown')):
        try:
            observations(bad,entries)
        except AssertionError:
            pass
        else:
            raise AssertionError('duplicate/future/unknown source accepted')
    assert len(allowed_urls('/static/2021/x_Race/',2021))==2
    for path,year in [('/static/2026/x_Race/',2026),('/static/2021/x_Sprint/',2021),('/static/2020/x_Race/',2021)]:
        try:
            allowed_urls(path,year)
        except AssertionError:
            pass
        else:
            raise AssertionError('unregistered endpoint accepted')


if __name__=='__main__':
    test_history()
    print('braking multi-season aggregation, strict prefix history, source identity and endpoint guards PASS')
