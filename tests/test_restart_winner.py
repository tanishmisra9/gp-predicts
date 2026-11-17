"""E207 restart history uses completed earlier events, with zero fallback."""
import numpy as np
import pandas as pd

from experiments.e207_restart_winner import NEW,add_history


def test_history():
    X=pd.DataFrame([dict(season=2018,round=k+1,k=k,code=d,team=('new' if k>2 else 'old'))
        for k in range(6) for d in ('a','b','rookie')])
    source=pd.DataFrame([dict(season=2018,round=k+1,restart=n,driver=d,gain=g)
        for k in range(6) for n in (1,2) for d,g in [('a',2*n),('b',-2*n)]])
    output=add_history(X,source)
    assert output.loc[X['round'].eq(1),NEW].eq(0).all().all()
    expected=3*.5**.1/(5+.5**.1)
    np.testing.assert_allclose(output.loc[X['round'].eq(2)&X.code.eq('a'),NEW],expected)
    assert output.loc[X.code.eq('rookie'),NEW].eq(0).all().all()
    assert add_history(X,source.iloc[:0])[NEW].eq(0).all().all()
    changed=source.copy(); changed.loc[changed['round'].ge(4),'gain']=9
    again=add_history(X,changed)
    np.testing.assert_array_equal(output.loc[X['round'].le(4),NEW],again.loc[X['round'].le(4),NEW])
    prefix=add_history(X,source[source['round'].lt(4)])
    np.testing.assert_array_equal(output.loc[X['round'].eq(4),NEW],prefix.loc[X['round'].eq(4),NEW])
    earlier=source.copy(); earlier.loc[earlier['round'].eq(1),'gain']=10
    assert not add_history(X,earlier).loc[X['round'].eq(4),NEW].equals(output.loc[X['round'].eq(4),NEW])
    for bad in (pd.concat([source,source.iloc[:1]]),source.assign(driver='unknown'),source.assign(gain=np.nan)):
        try:
            add_history(X,bad)
        except (AssertionError,pd.errors.MergeError):
            pass
        else:
            raise AssertionError('duplicate/unmatched/nonfinite source accepted')


if __name__=='__main__':
    test_history()
    print('restart per-event averaging, shrinkage/fallback, source keys and strict prior timing PASS')
