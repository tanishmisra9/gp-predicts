import numpy as np
import pandas as pd
from experiments.e251_roster_sensitivity import prior_roster,history_target,evaluate,score
from f1.data import DATA,load_results
from f1.features import build_features,pre_weekend_cols


def main():
    raw=load_results().loc[lambda d:d.season.le(2022)].copy()
    pace=pd.read_csv(DATA/"racepace_ff1.csv").loc[lambda d:d.season.le(2022)]
    assert prior_roster(raw,2022,1) is None
    prior=prior_roster(raw,2022,10)
    assert len(prior)==20
    target=history_target(raw,pace,2022,10);F=pre_weekend_cols(target)
    X=build_features(raw,pace)
    actual=X[X.season.eq(2022)&X["round"].eq(10)]
    assert set(actual.driver)==set(target.driver)
    a=actual.set_index("driver")[F].sort_index();b=target.set_index("driver")[F].sort_index()
    pd.testing.assert_frame_equal(a,b,check_dtype=False,atol=0,rtol=0)
    changed=raw.copy(); mask=changed.season.eq(2022)&changed["round"].ge(10)
    changed.loc[mask,"driver"]="unknown_after_cutoff";changed.loc[mask,"code"]="ZZZ";changed.loc[mask,"team"]="future_team"
    changed.loc[mask,"order"]=99;changed.loc[mask,"points"]=999
    cp=pace.copy();cp.loc[cp.season.eq(2022)&cp["round"].ge(10),"pace_gap"]=999
    again=history_target(changed,cp,2022,10)
    pd.testing.assert_frame_equal(target.set_index("driver")[F].sort_index(),again.set_index("driver")[F].sort_index(),check_dtype=False,atol=0,rtol=0)
    train=X[X.k.lt(target.k.iloc[0])&X.season.ge(2019)]
    original=score(train,target,F);mutated=score(train,again,F)
    for name in original:np.testing.assert_array_equal(original[name],mutated[name])
    changed=raw.copy();ids=changed.index[changed.season.eq(2022)&changed["round"].eq(9)]
    changed.loc[ids[0],"driver"]="earlier_replacement";changed.loc[ids[0],"code"]="EAR"
    assert "earlier_replacement" in set(history_target(changed,pace,2022,10).driver)
    try:history_target(raw,pace,2022,1)
    except ValueError:pass
    else:raise AssertionError("first round fabricated")
    prediction=pd.DataFrame(dict(season=[2022]*3,round=[10]*3,driver=["absent","real1","real2"],model=["test"]*3,policy=["prior"]*3,score=[0,1,2]))
    truth=pd.DataFrame(dict(season=[2022]*3,round=[10]*3,driver=["real1","real2","real3"],order=[1,2,3]))
    result=evaluate(prediction,truth).iloc[0]
    assert result.winner==0 and result.podium==2/3
    print("E251 actual-feature parity/current+future roster/labels/pace+scores invariant/earlier responsiveness/first-round/absent pick checks PASS")


if __name__=="__main__":main()
