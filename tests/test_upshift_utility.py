import numpy as np
import pandas as pd
from experiments.e245_upshift_utility import corrections


def main():
    rows=[dict(round=r,team=t,prediction=x,past_count=3,error=2*x) for r in range(1,7) for t,x in [("A",-1.),("B",1.)]]
    d=pd.DataFrame(rows)
    p=corrections(d)
    assert p[p["round"].le(3)].correction.eq(0).all()
    weights=np.array([.5**((4-r)/10) for r in range(1,4)])
    expected=4*weights.sum()/(1+2*weights.sum())
    assert np.allclose(p[p["round"].eq(4)].coefficient,expected)
    assert np.allclose(p.groupby("round").correction.sum(),0)
    changed=d.copy(); changed.loc[changed["round"].ge(4),"error"]=-100
    np.testing.assert_array_equal(corrections(changed).loc[p["round"].le(4),"correction"],p.loc[p["round"].le(4),"correction"])
    changed=d.copy(); changed.loc[changed["round"].eq(1),"error"]=-10*d.loc[d["round"].eq(1),"prediction"]
    assert not np.array_equal(corrections(changed).loc[p["round"].eq(4),"correction"],p.loc[p["round"].eq(4),"correction"])
    changed=d.copy(); changed["error"]=-d.error
    np.testing.assert_allclose(corrections(changed).correction,-p.correction)
    assert corrections(d.iloc[:0]).empty
    try:corrections(pd.concat([d,d.iloc[:1]]))
    except AssertionError:pass
    else:raise AssertionError("duplicate accepted")
    changed=d.copy(); changed["past_count"]=0
    assert corrections(changed).correction.eq(0).all()
    print("E245 hand coefficient/zero warmup/negative sign/strictpast labels/earlier sensitivity/empty/duplicate checks PASS")


if __name__=="__main__":main()
