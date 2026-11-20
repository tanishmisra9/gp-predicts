import pandas as pd
from experiments.e254_historical_links import links,qualified,topology


def main():
    rows=[dict(season=2014,round=r,driver=d,team="T",date="2014-01-01",q1=90.) for r in (1,2,3) for d in ("A","B")]
    d=pd.DataFrame(rows)
    observations,coverage=links(d,2014,2017)
    assert qualified(observations)=={("A","B")} and coverage["2014"]["fraction"]==1
    assert qualified(observations[:2])==set()
    c=d.copy();c.loc[0,"q1"]=0
    assert len(links(c,2014,2017)[0])==2
    future=d.copy();future["season"]=2020;future["date"]="2020-01-01";future["q1"]=1
    assert links(pd.concat([d,future],ignore_index=True),2014,2017)==links(d,2014,2017)
    c=d.copy();c.loc[0,"driver"]="EARLIER"
    assert links(c,2014,2017)[0]!=observations
    try:links(pd.concat([d,d.iloc[:1]],ignore_index=True),2014,2017)
    except AssertionError:pass
    else:raise AssertionError("duplicate accepted")
    base=topology({("A","B"),("C","D")},["A","B","C","D"])
    extended=topology({("A","B"),("C","D"),("B","C")},["A","B","C","D"])
    assert len(base["reference_connected_pairs"])==2 and len(extended["reference_connected_pairs"])==6
    assert topology(set(),["A","B"])["reference_pair_fraction"]==0
    assert topology(set(),[])["reference_pair_fraction"]==0
    print("E254 same-team/finiteQ1/3-event threshold/future exclusion/earlier response/duplicate/connectivity/isolate checks PASS")


if __name__=="__main__":main()
