"""Scalar histories and actual source/label timing mutations for E260."""
import math
from unittest.mock import patch
import numpy as np
import pandas as pd
from experiments.e168_winner import win_score
from experiments.e190_traffic_feasibility import cached_session
from experiments.e258_gear_ratio import windows,measure
from experiments.e260_ratio_winner import NEW,add_history
from f1.data import ROOT,DATA,load_results
from f1.features import build_features,pre_weekend_cols


def main():
    df=load_results().loc[lambda d:d.season.le(2022)]
    pace=pd.read_csv(DATA/"racepace_ff1.csv").loc[lambda d:d.season.le(2022)]
    X=build_features(df,pace).merge(df[["season","round","code","date"]],on=["season","round","code"],validate="one_to_one");base=pre_weekend_cols(X)
    table=pd.read_csv(ROOT/"results/e260_observations.csv",float_precision="round_trip")
    actual=add_history(X,table)
    saved=pd.read_csv(ROOT/"results/e260_features.csv",float_precision="round_trip")
    np.testing.assert_array_equal(actual[["season","round","code","k"]],saved[["season","round","code","k"]])
    np.testing.assert_allclose(actual[NEW],saved[NEW],atol=1e-15,rtol=1e-13)
    metadata={(r.season,r.round,r.code):(r.k,r.team,r.date) for r in X.itertuples()}
    records=[(*metadata[(r.season,r.round,r.code)],r.eligible,r.episodes) for r in table.itertuples() if pd.notna(r.eligible)]
    for k,current in X.groupby("k",sort=True):
        cutoff=current.date.iloc[0].normalize()-pd.Timedelta(days=3)
        past=[(2**(-(k-pk)/10),team,n,e) for pk,team,date,n,e in records if pk<k and date<cutoff]
        globalrate=(math.fsum(w*e for w,t,n,e in past)+.5)/(math.fsum(w*n for w,t,n,e in past)+1)
        for r in current.itertuples():
            expected=(math.fsum(w*e for w,t,n,e in past if t==r.team)+10000*globalrate)/(math.fsum(w*n for w,t,n,e in past if t==r.team)+10000)
            np.testing.assert_allclose(actual.loc[r.Index,NEW[0]],expected,atol=1e-15,rtol=1e-12)
    target=X[X.season.eq(2022)&X["round"].eq(10)];cutoff=target.k.iloc[0]
    keys={(r.season,r.round):r.k for r in X.itertuples()}
    future=table.apply(lambda r:keys[(r.season,r["round"])]>=cutoff,axis=1)
    altered=table.copy();valid=future&altered.eligible.notna();altered.loc[valid,"episodes"]=altered.loc[valid,"eligible"]
    changed=add_history(X,altered)
    np.testing.assert_array_equal(actual.loc[X.k.le(cutoff),NEW],changed.loc[X.k.le(cutoff),NEW])
    prefix=add_history(X,table[~future])
    np.testing.assert_array_equal(actual.loc[target.index,NEW],prefix.loc[target.index,NEW])
    training=X.k.lt(cutoff)&X.season.ge(2019)
    a=win_score(actual[training],actual.loc[target.index],base+NEW)
    b=win_score(changed[training],changed.loc[target.index].assign(gridn=-999,order=99,win=0,podium=0),base+NEW)
    np.testing.assert_array_equal(a,b)
    predictions=pd.read_csv(ROOT/"results/e260_inner_predictions.csv",float_precision="round_trip")
    wanted=predictions[predictions.season.eq(2022)&predictions["round"].eq(10)&predictions.model.eq("ratio_win")].set_index("driver")
    np.testing.assert_allclose(a,wanted.loc[target.driver,"score"],atol=1e-14,rtol=1e-14)
    earlier=table.copy();i=earlier[earlier.season.eq(2021)&earlier.eligible.notna()].index[-1];earlier.loc[i,"episodes"]+=10
    assert not add_history(X,earlier).loc[target.index,NEW].equals(actual.loc[target.index,NEW])
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True)
    with patch("requests.adapters.HTTPAdapter.send",side_effect=RuntimeError("mutation HTTP forbidden")):
        s=cached_session(fastf1,2022,10);laps=pd.DataFrame(s.laps);cars=fastf1._api.car_data(s.api_path)
    candidates=table[table.season.eq(2022)&table["round"].eq(10)&table.eligible.notna()].code.sort_values()
    assert len(candidates)>0
    driver=candidates.iloc[0];number=laps[laps.Driver.eq(driver)].DriverNumber.iloc[0]
    origin=max((c.Date-c.Time).max() for c in cars.values() if c.Time.notna().any()).round("ms")
    selected=windows(laps);car=cars[str(number)].copy();car["Speed"]=0
    counts,episodes=measure(car,selected[selected.Driver.eq(driver)],origin)
    assert counts["eligible"]==counts["episodes"]==0 and episodes.empty
    poisoned=table.copy();mask=poisoned.season.eq(2022)&poisoned["round"].eq(10)&poisoned.code.eq(driver)
    assert poisoned.loc[mask,"eligible"].notna().all()
    poisoned.loc[mask,["eligible","episodes"]]=np.nan
    np.testing.assert_array_equal(actual.loc[X.k.le(cutoff),NEW],add_history(X,poisoned).loc[X.k.le(cutoff),NEW])
    # Independently generate the seed260 one-label-per-earlier-race null and fit it.
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    tr=actual[training];rng=np.random.default_rng(260);labels=np.zeros(len(tr),dtype=int)
    for event in sorted(tr.k.unique()):
        positions=np.flatnonzero(tr.k.to_numpy()==event)
        labels[positions[rng.integers(len(positions))]]=1
    assert labels.sum()==tr.k.nunique()
    for event in tr.k.unique():assert labels[tr.k.to_numpy()==event].sum()==1
    model=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(C=.03,max_iter=500))
    weights=2.**((tr.k.to_numpy()-tr.k.max())/100)
    model.fit(tr[base+NEW],labels,logisticregression__sample_weight=weights)
    null=-model.predict_proba(actual.loc[target.index,base+NEW])[:,1]
    stored=predictions[predictions.season.eq(2022)&predictions["round"].eq(10)&predictions.model.eq("ratio_null")].set_index("driver")
    np.testing.assert_allclose(null,stored.loc[target.driver,"score"],atol=1e-14,rtol=1e-14)
    print("E260 scalar histories/saved features/actual source-future-prefix/earlier/fitted labels/raw-current mutation PASS")


if __name__=="__main__":main()
