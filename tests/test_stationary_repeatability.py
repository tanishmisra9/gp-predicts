"""E210 excludes rejected intervals and predicts using earlier events only."""
import numpy as np
import pandas as pd

from experiments.e191_traffic_repeatability import past_response
from experiments.e210_stationary_repeatability import team_table


def test_table():
    entries=pd.DataFrame([dict(season=2019,round=r,team=f't{t}',code=d)
        for r in range(1,7) for t in range(6) for d in (f'a{t}',f'b{t}')])
    source=entries.rename(columns={'code':'driver'}).drop(columns='team').assign(pitin_lap=10,usable=True)
    source['midpoint']=[3+int(d[1:])*.1 for d in source.driver]
    rejected=source.iloc[:1].assign(pitin_lap=20,midpoint=999,usable=False)
    table=team_table(source,entries)
    pd.testing.assert_frame_equal(team_table(pd.concat([source,rejected]),entries),table)
    assert table.centered_response.notna().all() and table.centered_half0.notna().all()
    np.testing.assert_allclose(table[table.team.eq('t0')].response,3)
    np.testing.assert_allclose(table[table.team.eq('t0')].centered_response,-.25)
    forecast=past_response(table)
    later=table.copy(); later.loc[later['round'].ge(4),'centered_response']=999
    again=past_response(later)
    np.testing.assert_array_equal(forecast.loc[forecast['round'].le(4),'prediction'],again.loc[again['round'].le(4),'prediction'])
    prior=table.copy(); prior.loc[prior['round'].eq(1),'centered_response']+=1
    assert past_response(prior).loc[table['round'].eq(4),'prediction'].ne(forecast.loc[table['round'].eq(4),'prediction']).all()
    for bad in (pd.concat([source,source.iloc[:1]]),source.assign(season=2026)):
        try:
            team_table(bad,entries)
        except AssertionError:
            pass
        else:
            raise AssertionError('duplicate/non2019 source accepted')
    missing=team_table(source[source.driver.str.startswith('a')],entries)
    assert missing.response.isna().all() and missing.centered_response.isna().all()


if __name__=='__main__':
    test_table()
    print('stationary team median/centering/rejected-row/key guards and forecast timing PASS')
