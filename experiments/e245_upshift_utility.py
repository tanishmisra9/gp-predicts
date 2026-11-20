"""E245 one prequential signed upshift correction; no winner model."""
import hashlib
import json

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.e191_traffic_repeatability import past_response
from f1.data import ROOT


FILES=("e244_team_measurements.csv","e194_forecasts_errors_scales.csv","e195_bias_forecasts.csv")


def assemble(table,errors,bias):
    for d in (table,errors,bias):assert d.season.eq(2019).all()
    assert not table.duplicated(["round","team"]).any()
    for d in (errors,bias):assert not d.duplicated(["round","driver"]).any()
    base=past_response(table)
    null=table.copy(); rng=np.random.default_rng(245)
    for _,ids in null[null.centered_response.notna()].groupby("round",sort=True).groups.items():
        null.loc[ids,"centered_response"]=rng.permutation(null.loc[ids,"centered_response"].to_numpy())
    null=past_response(null)
    out=base[["season","round","team","prediction","past_count"]].copy()
    out["null_prediction"]=null.prediction
    out["null_count"]=null.past_count
    for name in ("prediction","null_prediction"):
        out[name]-=out.groupby("round")[name].transform("mean")
    targets=errors.groupby(["round","team"],sort=True).agg(error=("error","mean"),
        mean_forecast=("mean_forecast","mean"),y_pace=("y_pace","mean")).reset_index()
    group=targets.groupby("round").error
    targets["error"]=(targets.error-group.transform("mean")).where(group.transform("count").ge(6))
    control=bias.groupby(["round","team"],sort=True).correction.mean().rename("bias_correction").reset_index()
    out=out.merge(targets,on=["round","team"],how="left",validate="one_to_one")
    out=out.merge(control,on=["round","team"],how="left",validate="one_to_one")
    assert len(out)==len(table)
    return out


def corrections(frame,x="prediction",count="past_count"):
    out=frame.copy(); out["correction"]=0.; out["coefficient"]=0.;out["training_races"]=0
    assert not out.duplicated(["round","team"]).any()
    for round_,g in out.groupby("round",sort=True):
        past=out[out["round"].lt(round_)&out.error.notna()&out[count].ge(3)]
        n=past["round"].nunique(); out.loc[g.index,"training_races"]=n
        if n<3:continue
        weight=.5**((round_-past["round"])/10)
        beta=float((weight*past[x]*past.error).sum()/(1+(weight*past[x]**2).sum()))
        out.loc[g.index,"coefficient"]=beta
        eligible=g[count].ge(3)
        values=beta*g.loc[eligible,x]
        out.loc[values.index,"correction"]=values-values.mean()
    return out


def compare(a,b):
    value,ci,p=W.paired_boot(a,b)
    return dict(gain=float(value),ci=ci.tolist(),p_nonpositive=float(p))


def summarize(scored):
    valid=scored[scored.error.notna()].copy()
    assert valid.bias_correction.notna().all()
    valid["base_loss"]=valid.error**2
    valid["candidate_loss"]=(valid.error-valid.correction)**2
    valid["null_loss"]=(valid.error-valid.null_correction)**2
    valid["bias_loss"]=(valid.error-valid.bias_correction)**2
    races=valid.groupby("round",sort=True)[["base_loss","candidate_loss","null_loss","bias_loss"]].mean()
    ranks=[]
    for round_,g in valid.groupby("round",sort=True):
        if len(g)<6:continue
        base=g.mean_forecast.corr(g.y_pace,method="spearman")
        candidate=(g.mean_forecast+g.correction).corr(g.y_pace,method="spearman")
        ranks.append(dict(round=round_,base=base,candidate=candidate))
    ranks=pd.DataFrame(ranks,columns=["round","base","candidate"])
    baseline=compare(races.base_loss,races.candidate_loss)
    bias=compare(races.bias_loss,races.candidate_loss)
    null=compare(races.null_loss,races.candidate_loss)
    order=compare(ranks.candidate,ranks.base) if len(ranks) else dict(gain=0.,ci=[0.,0.],p_nonpositive=1.)
    learned=valid[valid.training_races.ge(3)&valid.past_count.ge(3)]
    reduction=baseline["gain"]/races.base_loss.mean() if len(races) and races.base_loss.mean()>0 else 0.
    gate=bool(len(valid)>=120 and len(races)>=12 and len(learned)>=100 and learned["round"].nunique()>=10 and
              reduction>=.02 and all(r["gain"]>0 and r["ci"][0]>0 for r in (baseline,bias,null,order)))
    return dict(rows=len(valid),races=len(races),learned_rows=len(learned),learned_races=int(learned["round"].nunique()),
                race_mean_losses=races.mean().to_dict(),baseline=baseline,bias=bias,shuffled=null,
                relative_mse_reduction=float(reduction),pace_order=order,gate=gate,winner_fit=False),valid,races,ranks


def main():
    table=pd.read_csv(ROOT/"results"/FILES[0],float_precision="round_trip")
    errors=pd.read_csv(ROOT/"results"/FILES[1],float_precision="round_trip").loc[lambda d:d.season.eq(2019)]
    bias=pd.read_csv(ROOT/"results"/FILES[2],float_precision="round_trip").loc[lambda d:d.season.eq(2019)]
    assert len(table)==210 and table["round"].nunique()==21
    frame=assemble(table,errors,bias)
    scored=corrections(frame)
    shuffled=corrections(frame,"null_prediction","null_count")
    scored["null_correction"]=shuffled.correction
    scored["null_coefficient"]=shuffled.coefficient
    summary,valid,races,ranks=summarize(scored)
    summary["hashes"]={n:hashlib.sha256((ROOT/"results"/n).read_bytes()).hexdigest() for n in FILES}
    for name,d in (("forecasts",scored),("scored",valid),("race_losses",races.reset_index()),("pace_order",ranks)):
        d.to_csv(ROOT/f"results/e245_{name}.csv",index=False)
    (ROOT/"results/e245_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
