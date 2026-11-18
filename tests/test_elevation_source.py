"""E218 constant/driver-offset Z is not a spatial elevation profile."""
import numpy as np
import pandas as pd

from experiments.e218_elevation_source import COLS,evaluate,profiles


def test_shape():
    rows=[]
    for driver in range(10):
        for cell in range(50):
            for subset in ('full','half0','half1'):
                rows.append(dict(driver=str(driver),cellx=cell,celly=0,subset=subset,
                    height=20*np.sin(cell/10)+driver,n=10))
    table=pd.DataFrame(rows)[COLS]
    assert evaluate(table)['shape_gate']
    assert not evaluate(table.assign(height=0.))['shape_gate']
    assert not evaluate(table.assign(height=table.driver.astype(float)))['shape_gate']
    assert not evaluate(table[table.driver.ne('0')])['shape_gate']
    assert not evaluate(table.iloc[:0])['shape_gate']
    bad=table.copy(); bad.loc[bad.subset.eq('half1'),'height']*=-1
    assert not evaluate(bad)['shape_gate']
    try:
        evaluate(pd.concat([table,table.iloc[:1]]))
    except AssertionError:
        pass
    else:
        raise AssertionError('duplicate profile accepted')


def test_projection():
    n=40; t=np.arange(n)
    raw=pd.DataFrame(dict(Date=pd.Timestamp('2022-01-01')+pd.to_timedelta(t,unit='s'),
        Time=pd.to_timedelta(t,unit='s'),Status='OnTrack',X=-501,Y=0,Z=123))
    table,counts=profiles({'1':raw})
    assert table.cellx.eq(-2).all() and table.height.eq(12.3).all()
    assert table.n.tolist()==[40,20,20] and counts['observed']==40
    raw.loc[0,'Time']=pd.NaT
    assert profiles({'1':raw})[1]['observed']==39
    assert profiles({'1':raw.assign(Status='OffTrack')})[0].empty
    assert profiles({})[0].empty
    assert not evaluate(profiles({})[0])['shape_gate']


if __name__=='__main__':
    test_shape(); test_projection()
    print('elevation spatial versus constant/driver-offset Z, parity, units, cells, padding and support gates PASS')
