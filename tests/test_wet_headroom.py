"""Wet-tyre measurement support and boundaries; no network or winner model."""
import pandas as pd
from experiments.e225_wet_headroom import wet_label


def test_label():
    rows=[dict(Driver=str(d),LapNumber=l,Compound='SOFT') for d in range(10) for l in range(1,5)]
    x=pd.DataFrame(rows); assert wet_label(x)['wet']==False
    a=x.copy(); a.loc[a.Driver.astype(int).lt(5)&a.LapNumber.le(2),'Compound']='INTERMEDIATE'
    assert wet_label(a)['wet']==True and wet_label(a)['wet_drivers_two_laps']==5
    b=a.copy(); b.loc[b.Driver.eq('4')&b.LapNumber.eq(2),'Compound']='SOFT'
    assert wet_label(b)['wet']==False
    assert not wet_label(a[a.Driver.ne('9')])['available']
    missing=x.copy(); missing.loc[:4,'Compound']='UNKNOWN'
    assert not wet_label(missing)['available']
    assert not wet_label(x.iloc[:0])['available']
    assert wet_label(a.assign(order=999,grid=999,rain_act=999))==wet_label(a)
    try:wet_label(pd.concat([x,x.iloc[:1]],ignore_index=True))
    except AssertionError:pass
    else:raise AssertionError('duplicate lap accepted')


if __name__=='__main__':
    test_label(); print('Wet tyre label support/boundary tests ok')
