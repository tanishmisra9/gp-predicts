"""E245 scalar historical fits, labels, matched losses and bootstrap audit."""
import hashlib
import json
import math
from statistics import mean

import numpy as np
import pandas as pd

from experiments.e245_upshift_utility import assemble,corrections
from tests.test_upshift_forecast_artifact import forecast
from f1.data import ROOT


def main():
    report=json.loads((ROOT/"results/e245_summary.json").read_text())
    for name,digest in report["hashes"].items():
        assert hashlib.sha256((ROOT/"results"/name).read_bytes()).hexdigest()==digest
    table=pd.read_csv(ROOT/"results/e244_team_measurements.csv",float_precision="round_trip")
    errors=pd.read_csv(ROOT/"results/e194_forecasts_errors_scales.csv",float_precision="round_trip").loc[lambda d:d.season.eq(2019)]
    bias=pd.read_csv(ROOT/"results/e195_bias_forecasts.csv",float_precision="round_trip").loc[lambda d:d.season.eq(2019)]
    saved=pd.read_csv(ROOT/"results/e245_forecasts.csv",float_precision="round_trip")
    observed=pd.read_csv(ROOT/"results/e245_scored.csv",float_precision="round_trip")
    racefile=pd.read_csv(ROOT/"results/e245_race_losses.csv",float_precision="round_trip")
    rankfile=pd.read_csv(ROOT/"results/e245_pace_order.csv",float_precision="round_trip")
    for _,g in errors.groupby("round",sort=True):
        good=g[g.y_pace.notna()]
        center=mean([r.y_pace-r.mean_forecast for r in good.itertuples()])
        for r in good.itertuples():assert abs(r.error-(r.y_pace-r.mean_forecast-center))<1e-12
    # Independently reconstruct the old E195 comparator from earlier team errors.
    history=[]
    for (k,team),g in errors.groupby(["k","team"],sort=True):
        values=g.error.dropna().tolist()
        if values:history.append((k,team,mean(values)))
    for k,g in bias.groupby("k",sort=True):
        raw=[]
        for r in g.itertuples():
            h=[(2**(-(k-pk)/20),v) for pk,t,v in history if pk<k and t==r.team]
            raw.append(math.fsum(w*v for w,v in h)/(8+math.fsum(w for w,v in h)))
        np.testing.assert_allclose(np.array(raw)-mean(raw),g.correction,atol=1e-12)
    expected=table[["season","round","team"]].copy()
    expected["prediction"],expected["past_count"]=forecast(table)
    perm=table.copy();rng=np.random.default_rng(245)
    for _,ids in perm[perm.centered_response.notna()].groupby("round",sort=True).groups.items():
        perm.loc[ids,"centered_response"]=rng.permutation(perm.loc[ids,"centered_response"].to_numpy())
    expected["null_prediction"],expected["null_count"]=forecast(perm)
    for _,ids in expected.groupby("round",sort=True).groups.items():
        for col in ("prediction","null_prediction"):
            expected.loc[ids,col]-=mean(expected.loc[ids,col])
    records=[]
    for r in expected.itertuples():
        row=r._asdict();row.pop("Index")
        g=errors[errors["round"].eq(r.round)&errors.team.eq(r.team)]
        for col in ("error","mean_forecast","y_pace"):
            vals=g[col].dropna().tolist();row[col]=mean(vals) if vals else np.nan
        vals=bias[bias["round"].eq(r.round)&bias.team.eq(r.team)].correction.dropna().tolist()
        row["bias_correction"]=mean(vals) if vals else np.nan
        records.append(row)
    expected=pd.DataFrame(records)
    for _,ids in expected.groupby("round",sort=True).groups.items():
        vals=expected.loc[ids,"error"].dropna().tolist()
        expected.loc[ids,"error"]=expected.loc[ids,"error"]-mean(vals) if len(vals)>=6 else np.nan
    def fit(x,count):
        predictions=[];betas=[];training=[]
        for r in expected.itertuples():
            past=[p for p in expected.itertuples() if p.round<r.round and pd.notna(p.error) and getattr(p,count)>=3]
            n=len({p.round for p in past});training.append(n)
            weights=[2**(-(r.round-p.round)/10) for p in past]
            beta=math.fsum(w*getattr(p,x)*p.error for w,p in zip(weights,past))/(1+math.fsum(w*getattr(p,x)**2 for w,p in zip(weights,past))) if n>=3 else 0.
            betas.append(beta);predictions.append(beta*getattr(r,x) if getattr(r,count)>=3 else 0.)
        predictions=np.array(predictions)
        for _,g in expected.groupby("round",sort=True):
            ids=g[g[count].ge(3)].index
            if len(ids):predictions[ids]-=mean(predictions[ids])
        return predictions,np.array(betas),np.array(training)
    p,b,n=fit("prediction","past_count")
    q,c,_=fit("null_prediction","null_count")
    expected["correction"]=p;expected["coefficient"]=b;expected["training_races"]=n
    expected["null_correction"]=q;expected["null_coefficient"]=c
    pd.testing.assert_frame_equal(saved,expected,check_dtype=False,atol=1e-12,rtol=1e-12)
    valid=expected[expected.error.notna()].copy()
    for col,correction in [("base_loss",0),("candidate_loss",valid.correction),("null_loss",valid.null_correction),("bias_loss",valid.bias_correction)]:
        valid[col]=(valid.error-correction)**2
    pd.testing.assert_frame_equal(observed,valid.reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=1e-12)
    races=[];ranks=[]
    def rank(values):
        return np.array([sum(v<x for v in values)+(sum(v==x for v in values)+1)/2 for x in values])
    for round_,g in valid.groupby("round",sort=True):
        races.append(dict(round=round_,**{col:mean(g[col]) for col in ("base_loss","candidate_loss","null_loss","bias_loss")}))
        if len(g)>=6:
            target=rank(g.y_pace.to_numpy())
            ranks.append(dict(round=round_,base=np.corrcoef(rank(g.mean_forecast.to_numpy()),target)[0,1],
                              candidate=np.corrcoef(rank((g.mean_forecast+g.correction).to_numpy()),target)[0,1]))
    races=pd.DataFrame(races);ranks=pd.DataFrame(ranks)
    pd.testing.assert_frame_equal(racefile,races,check_dtype=False,atol=1e-12,rtol=1e-12)
    pd.testing.assert_frame_equal(rankfile,ranks,check_dtype=False,atol=1e-12,rtol=1e-12)
    def check(a,b,expected):
        delta=np.asarray(a)-np.asarray(b)
        rng=np.random.default_rng(0)
        indices=rng.integers(len(delta),size=(5000,len(delta)))
        samples=np.array([math.fsum(delta[i] for i in ids)/len(ids) for ids in indices])
        assert abs(mean(delta)-expected["gain"])<1e-12
        np.testing.assert_allclose(np.quantile(samples,[.025,.975]),expected["ci"],atol=1e-12)
        assert (samples<=0).mean()==expected["p_nonpositive"]
    for col,name in [("base_loss","baseline"),("bias_loss","bias"),("null_loss","shuffled")]:
        check(races[col],races.candidate_loss,report[name])
    check(ranks.candidate,ranks.base,report["pace_order"])
    for col in ("base_loss","candidate_loss","null_loss","bias_loss"):
        assert abs(mean(races[col])-report["race_mean_losses"][col])<1e-12
    learned=valid[valid.training_races.ge(3)&valid.past_count.ge(3)]
    assert [len(valid),len(races),len(learned),learned["round"].nunique()]==[report[k] for k in ("rows","races","learned_rows","learned_races")]
    reduction=report["baseline"]["gain"]/mean(races.base_loss)
    assert abs(reduction-report["relative_mse_reduction"])<1e-12
    gate=bool(len(valid)>=120 and len(races)>=12 and len(learned)>=100 and learned["round"].nunique()>=10 and reduction>=.02 and
              all(report[k]["gain"]>0 and report[k]["ci"][0]>0 for k in ("baseline","bias","shuffled","pace_order")))
    assert report["gate"]==gate and not report["winner_fit"]
    # Mutation checks exercise the actual new code separately from reconstruction.
    for label in (False,True):
        changed=table.copy();changed_errors=errors.copy()
        if label:changed_errors.loc[changed_errors["round"].ge(10),"error"]+=1000
        else:changed.loc[changed["round"].ge(10),"centered_response"]+=1000
        frame=assemble(changed,changed_errors,bias)
        actual=corrections(frame)
        np.testing.assert_array_equal(actual.loc[actual["round"].le(10),"correction"],saved.loc[saved["round"].le(10),"correction"])
        actual=corrections(frame,"null_prediction","null_count")
        np.testing.assert_array_equal(actual.loc[actual["round"].le(10),"correction"],saved.loc[saved["round"].le(10),"null_correction"])
    changed=table.copy();changed.loc[changed["round"].eq(1)&changed.team.eq(changed.team.iloc[0]),"centered_response"]+=2
    actual=corrections(assemble(changed,errors,bias))
    assert actual.loc[actual["round"].eq(10),"correction"].ne(saved.loc[saved["round"].eq(10),"correction"]).any()
    changed=errors.copy();changed.loc[changed["round"].eq(6)&changed.team.eq(changed.team.iloc[0]),"error"]+=20
    actual=corrections(assemble(table,changed,bias))
    assert actual.loc[actual["round"].eq(10),"correction"].ne(saved.loc[saved["round"].eq(10),"correction"]).any()
    print("E245 independent raw labels/E195 control/scalar histories/coefficients/null/loss/ranks/bootstrap/gate/hash/source and label mutations PASS")


if __name__=="__main__":main()
