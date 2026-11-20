import numpy as np
import pandas as pd
from experiments.e255_historical_link_transfer import fit,direct,shuffled


def data():
    source=pd.DataFrame([dict(season=2019,round=i,team="T",a="A",b="B",date=pd.Timestamp(f"2019-01-{i:02d}"),gap=1.) for i in (1,2,3)])
    current=pd.DataFrame([dict(season=2020,round=1,team="T",a="A",b="B",date=pd.Timestamp("2020-01-05"),gap=99.)])
    return source,current


def main():
    source,current=data()
    p,state=fit(source,current,False)
    cutoff=current.date.iloc[0]-pd.Timedelta(days=3)
    w=.5**((cutoff-source.date).dt.total_seconds().to_numpy()/86400/730.5)
    np.testing.assert_allclose(p,2*w.sum()/(1+2*w.sum()),atol=1e-14)
    np.testing.assert_allclose(direct(source,current)[0],w.sum()/(1+w.sum()),atol=1e-14)
    changed=pd.concat([source,current],ignore_index=True)
    changed.loc[changed.season.eq(2020),"gap"]=-999
    np.testing.assert_array_equal(p,fit(changed,current,False)[0])
    current2=current.copy();current2["gap"]=-1000
    np.testing.assert_array_equal(p,fit(source,current2,False)[0])
    changed=source.copy();changed.loc[0,"gap"]=-10
    assert fit(changed,current,False)[0][0]!=p[0]
    assert fit(source.iloc[:2],current,False)[0][0]==0
    assert fit(source.iloc[:0],current,True)[0][0]==0
    old=source.copy();old["season"]=2017;old["date"]-=pd.Timedelta(days=730)
    assert fit(old,current,True)[0][0]>0 and fit(old,current,False)[0][0]==0
    future=pd.concat([source,current],ignore_index=True)
    np.testing.assert_array_equal(fit(shuffled(future),current,True)[0],fit(shuffled(source),current,True)[0])
    print("E255 hand ridge/direct/strictpast target+source/shuffle/earlier response/3-event/empty/old-only checks PASS")


if __name__=="__main__":main()
