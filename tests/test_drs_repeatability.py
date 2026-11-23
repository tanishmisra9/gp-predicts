import io
from unittest.mock import patch
import numpy as np
import pandas as pd
from experiments.e263_drs_repeatability import acceleration,lap_components,aggregate,evaluate,correlation


def main():
    origin=pd.Timestamp("2019-01-01");t=np.array([0,.2,.4,.7,1.])
    frame=pd.DataFrame(dict(Date=origin+pd.to_timedelta(t,unit="s"),Time=pd.to_timedelta(t,unit="s"),Speed=220+3.6*2*t,Throttle=100,Brake=False,nGear=6,DRS=12))
    a,span=acceleration(frame,2,1);assert abs(a-2)<1e-12 and span==1
    assert abs(acceleration(frame.assign(Speed=frame.Speed+100),2,1)[0]-a)<1e-12
    assert acceleration(frame.assign(Speed=250),2,1)[0]==0
    for edited in (frame.assign(Time=pd.NaT),frame.assign(DRS=9),frame.assign(Speed=np.nan),frame.assign(nGear=5.5),frame.assign(Brake=True)):
        try:acceleration(edited,2,1)
        except AssertionError:pass
        else:raise AssertionError("invalid window accepted")
    duplicate=frame.copy();duplicate.loc[2,"Date"]=duplicate.loc[1,"Date"]
    try:acceleration(duplicate,2,1)
    except AssertionError:pass
    else:raise AssertionError("duplicate Date accepted")
    for index in (0,3):
        try:acceleration(frame,index,1)
        except AssertionError:pass
        else:raise AssertionError("window bound accepted")
    chain=pd.DataFrame(dict(on_lap=[2,3,7],off_lap=[3,4,8]))
    np.testing.assert_array_equal(lap_components(chain),[0,0,1])
    perm=chain.iloc[::-1];np.testing.assert_array_equal(lap_components(perm),[1,0,0])
    pairs=pd.DataFrame(dict(season=2019,round=5,driver="A",on_index=[1,2,3,4],off_index=[11,12,13,14],on_lap=[2,2,3,7],off_lap=[3,3,4,8],contrast=[2.,4.,9.,10.]))
    support=pd.DataFrame(dict(season=[2019],round=[5],driver=["A"],supported=[True]))
    tagged,lp,comp,table=aggregate(pairs,support)
    _,_,_,permuted=aggregate(pairs.iloc[::-1],support)
    pd.testing.assert_frame_equal(table,permuted)
    try:aggregate(pairs.assign(contrast=np.nan),support)
    except AssertionError:pass
    else:raise AssertionError("missing contrast accepted")
    assert table.full.iloc[0]==8 and table.half0.iloc[0]==6 and table.half1.iloc[0]==10 and not table.repeat_supported.iloc[0]
    assert set(tagged[tagged.half.eq(0)].on_lap)|set(tagged[tagged.half.eq(0)].off_lap)=={2,3,4}
    rows=[]
    for start in (2,10):
        for i in range(6):rows.append(dict(season=2019,round=5,driver="A",on_index=start*100+i,off_index=start*100+50+i,on_lap=start+i%3,off_lap=start+i%3+1,contrast=1.))
    _,_,_,enough=aggregate(pd.DataFrame(rows),support);assert enough.repeat_supported.iloc[0]
    _,_,_,short=aggregate(pd.DataFrame(rows).iloc[2:],support);assert not short.repeat_supported.iloc[0]
    no_pairs=pairs.iloc[:0];_,_,_,empty=aggregate(no_pairs,support)
    assert pd.isna(empty.full.iloc[0]) and not empty.repeat_supported.iloc[0]
    positive=pd.DataFrame([dict(season=2019,round=r,driver=str(i),half0=float(i),half1=float(2*i+r),repeat_supported=True) for r in (5,12) for i in range(6)])
    result,_=evaluate(positive);assert result["gate"] and result["bootstrap_finite"]==4000 and result["identity_null_finite"]==4000
    negative=positive.assign(half1=-positive.half1);result,_=evaluate(negative);assert not result["gate"]
    constant=positive.assign(half1=1.);result,_=evaluate(constant);assert not result["gate"] and result["bootstrap_finite"]==0 and result["bootstrap_ci"] is None
    result,_=evaluate(positive[positive.driver.ne("0")]);assert not result["coverage_gate"] and result["bootstrap_finite"]==0
    with patch("numpy.random.default_rng",side_effect=AssertionError("coverage-failed branch must not resample")):
        _,empty_score=evaluate(positive[positive.driver.ne("0")])
    assert pd.read_csv(io.StringIO(empty_score.to_csv(index=False))).empty
    assert {"x","y","half0","half1"}.issubset(empty_score.columns)
    assert correlation([1,1],[2,3]) is None
    print("E263 OLS/intercept/bounds/guards/graph/inputorder/lap disjointness/weights/missing/support/bootstrap/null tests PASS")


if __name__=="__main__":main()
