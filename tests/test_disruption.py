"""E205 source windows and strictly earlier team aggregation."""
import numpy as np
import pandas as pd
import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiments.e205_disruption_winner import NEW,add_history,disruptions
from tests.test_warmup import fixture


def test_source_history():
    x=fixture()
    zero=disruptions(x); assert zero.eligible.iloc[0]==3 and zero.disruptions.iloc[0]==0
    x.loc[5,'LapTime']=pd.Timedelta(seconds=93)
    measured=disruptions(x); assert measured.disruptions.iloc[0]==1
    pd.testing.assert_frame_equal(disruptions(x.assign(order=99,grid=-1)),measured)
    for name,value in [('TrackStatus','4'),('IsAccurate',False),('Stint',3),('TyreLife',99),('Compound','WET')]:
        bad=x.copy(); bad.loc[5,name]=value
        assert disruptions(bad).disruptions.iloc[0]==0
    assert disruptions(x.iloc[:0]).empty
    try:
        disruptions(pd.concat([x,x.iloc[:1]]))
    except AssertionError:
        pass
    else:
        raise AssertionError('duplicate source accepted')
    X=pd.DataFrame([dict(season=2018,round=k+1,k=k,code=d,team=t)
        for k in range(6) for d,t in [('a','t1'),('b','t2')]])
    source=X[['season','round','code']].assign(eligible=100,disruptions=1)
    output=add_history(X,source)
    assert output[NEW].notna().all().all() and output[NEW].ge(0).all().all()
    changed=source.copy(); changed.loc[changed['round'].ge(4),'disruptions']=90
    again=add_history(X,changed)
    np.testing.assert_array_equal(output.loc[X['round'].le(4),NEW],again.loc[X['round'].le(4),NEW])
    changed=source.copy(); changed.loc[(changed['round']==1)&changed.code.eq('a'),'disruptions']=50
    earlier=add_history(X,changed)
    assert earlier.loc[X['round'].eq(4),NEW].ne(output.loc[X['round'].eq(4),NEW]).any().any()
    target=X[X['round'].eq(4)]
    prefix=add_history(X,source[source['round'].lt(4)])
    np.testing.assert_array_equal(output.loc[target.index,NEW],prefix.loc[target.index,NEW])
    empty=add_history(X,source.iloc[:0]); assert empty[NEW].eq(.5).all().all()


def test_unloaded_session():
    import fastf1
    from fastf1.exceptions import DataNotLoadedError
    import experiments.e205_disruption_winner as experiment
    entries=pd.DataFrame([dict(season=s,round=r,code='a') for s,n in [(2018,21),(2019,21),(2020,17),(2021,22),(2022,22)] for r in range(1,n+1)])
    class Session:
        def __init__(self,missing):
            self.missing=missing
        @property
        def laps(self):
            if self.missing:
                raise DataNotLoadedError('synthetic unloaded cache')
            return fixture()
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory); (root/'results').mkdir()
        with patch.object(experiment,'ROOT',root),patch.object(experiment,'load_results',return_value=entries),patch.object(experiment,'cached_session',side_effect=lambda _,s,r:Session(s==2018 and r==14)),patch.object(fastf1.Cache,'offline_mode'),contextlib.redirect_stdout(io.StringIO()):
            source=experiment.extract()
        summary=json.loads((root/'results/e205_source_summary.json').read_text())
        assert len(source)==102 and summary['coverage_gate']
        missing=[r for r in summary['events'] if not r['available']]
        assert len(missing)==1 and missing[0]['error_type']=='DataNotLoadedError'


if __name__=='__main__':
    test_source_history()
    test_unloaded_session()
    print('disruption windows, neutralization/stint/tyre exclusions, metadata and history timing PASS')
