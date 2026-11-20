import numpy as np
import pandas as pd
from experiments.e260_ratio_winner import NEW,add_history,observations


def main():
    X=pd.DataFrame([dict(season=2018,round=k+1,k=k,code=c,team=t,date=pd.Timestamp("2018-03-25")+pd.Timedelta(days=7*k))
                    for k in range(4) for c,t in (("A","T"),("B","U"))])
    source=X[["season","round","code"]].assign(eligible=1000.,episodes=1.)
    actual=add_history(X,source)
    assert actual.loc[X.k.eq(0),NEW].eq(.5).all().all()
    w=2**(-.1);g=(2*w+.5)/(2000*w+1)
    expected=(w+10000*g)/(1000*w+10000)
    np.testing.assert_allclose(actual.loc[X.k.eq(1),NEW],expected,atol=1e-15)
    assert add_history(X,source.iloc[:0])[NEW].eq(.5).all().all()
    edited=source.copy();edited.loc[edited["round"].ge(3),"episodes"]=900
    np.testing.assert_array_equal(actual.loc[X.k.le(2),NEW],add_history(X,edited).loc[X.k.le(2),NEW])
    np.testing.assert_array_equal(actual.loc[X.k.eq(2),NEW],add_history(X,source[source["round"]<3]).loc[X.k.eq(2),NEW])
    edited=source.copy();edited.loc[0,"episodes"]=9
    assert not actual.loc[X.k.eq(2),NEW].equals(add_history(X,edited).loc[X.k.eq(2),NEW])
    for bad in (pd.concat([source,source.iloc[:1]]),source.assign(episodes=-1),source.assign(episodes=1001),source.assign(season=2026)):
        try:add_history(X,bad)
        except AssertionError:pass
        else:raise AssertionError("bad source accepted")
    reports=[dict(season=2018,round=1,rows=[dict(driver="A",available=True,eligible=100,episodes=1,supported_gears=2),
                                                        dict(driver="B",available=False)])]
    table=observations(reports,X[["season","round","code"]])
    assert table.eligible.notna().sum()==1 and table.episodes.sum()==1
    # Exercise the actual builder boundary: date is calendar metadata, not a feature.
    from f1.data import load_results
    from f1.features import build_features,pre_weekend_cols
    df=load_results().loc[lambda d:d.season.le(2022)]
    raw=build_features(df)
    assert "date" not in raw
    dated=raw.merge(df[["season","round","code","date"]],on=["season","round","code"],validate="one_to_one")
    assert pre_weekend_cols(raw)==pre_weekend_cols(dated) and len(raw)==len(dated)
    assert dated.date.notna().all()
    print("E260 hand shrinkage/empty/current-future/prefix/prior/invalid/missing mapping tests PASS")


if __name__=="__main__":main()
