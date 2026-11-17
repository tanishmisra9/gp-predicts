"""E207 restart history uses completed earlier events, with zero fallback."""
import numpy as np
import pandas as pd
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from experiments.e207_restart_winner import NEW,add_history
import experiments.e207_restart_winner as experiment


def test_history():
    X=pd.DataFrame([dict(season=2018,round=k+1,k=k,code=d,team=('new' if k>2 else 'old'))
        for k in range(6) for d in ('a','b','rookie')])
    source=pd.DataFrame([dict(season=2018,round=k+1,restart=n,driver=d,gain=g)
        for k in range(6) for n in (1,2) for d,g in [('a',2*n),('b',-2*n)]])
    output=add_history(X,source)
    pd.testing.assert_frame_equal(output,add_history(X,source.assign(gain=source.gain.astype(object))))
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


def test_bad_timeline():
    flags=pd.DataFrame(dict(Time=pd.to_timedelta([1,1],unit='s'),Status=['4','1']))
    session=SimpleNamespace(laps=pd.DataFrame(),track_status=flags)
    entries=pd.DataFrame(dict(season=[2019],round=[2]))
    with TemporaryDirectory() as directory:
        root=Path(directory); (root/'results').mkdir()
        with patch.object(experiment,'ROOT',root),patch.object(experiment,'load_results',return_value=entries),patch.object(experiment,'cached_session',return_value=session),contextlib.redirect_stdout(io.StringIO()):
            try:
                experiment.extract()
            except AssertionError:
                pass
            else:
                raise AssertionError('unavailable source passed coverage')
        report=json.loads((root/'results/e207_source_summary.json').read_text())
        assert report['available']==0 and not report['coverage_gate']
        assert report['sessions'][0]['error_type']=='ValueError'


if __name__=='__main__':
    test_history()
    test_bad_timeline()
    print('restart per-event averaging, shrinkage/fallback, source keys and strict prior timing PASS')
