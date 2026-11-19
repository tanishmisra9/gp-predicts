"""E235 observation identity and strictly earlier multi-year driver history."""
import numpy as np
import pandas as pd
from experiments.e235_launch_winner import NEW,add_history,observations


def test_history():
    X=pd.DataFrame([dict(season=2018+k//4,round=k%4+1,k=k,code=d) for k in range(8) for d in ('A','B')])
    reports=[dict(season=2018+k//4,round=k%4+1,gate=True,rows=[dict(driver=d,supported=True,relative20=v) for d,v in [('A',.1),('B',-.1)]]) for k in range(8)]
    table=observations(reports,X[['season','round','code']]); base=add_history(X,table)
    np.testing.assert_allclose(base.loc[X.k.eq(1)&X.code.eq('A'),NEW],.1*.5**.1/(5+.5**.1))
    altered=table.copy(); altered.loc[altered.season.eq(2019),'response']=99
    changed=add_history(X,altered)
    np.testing.assert_array_equal(base.loc[X.k.le(4),NEW],changed.loc[X.k.le(4),NEW])
    prefix=add_history(X,table[table.season.eq(2018)])
    np.testing.assert_array_equal(base.loc[X.k.eq(4),NEW],prefix.loc[X.k.eq(4),NEW])
    earlier=table.copy(); earlier.loc[0,'response']=1
    assert not add_history(X,earlier).loc[X.k.eq(4),NEW].equals(base.loc[X.k.eq(4),NEW])
    assert add_history(X,table.iloc[:0])[NEW].eq(0).all().all()
    assert observations([],X[['season','round','code']]).response.isna().all()
    for bad in (pd.concat([table,table.iloc[:1]]),table.assign(season=2026)):
        try: add_history(X,bad)
        except AssertionError: pass
        else: raise AssertionError('source identity accepted')
    for bad in ([reports[0],reports[0]],[dict(season=2026,round=1,gate=False,rows=[])],[dict(season=2018,round=1,gate=True,rows=[dict(driver='Z',supported=True,relative20=0)])]):
        try: observations(bad,X[['season','round','code']])
        except AssertionError: pass
        else: raise AssertionError('invalid report accepted')
    print('E235 prior-only multi-year history/empty/missing/identity/source guards pass')


if __name__=='__main__':
    test_history()
