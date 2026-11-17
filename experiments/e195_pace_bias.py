"""E195: one fixed signed prequential team-bias correction, no winner model."""
import json

import numpy as np
import pandas as pd

from experiments.e194_pace_error_scale import clustered
from f1.data import ROOT


def bias_predictions(entries,observations,shuffled=False):
    source=observations.sort_values(['k','team']).reset_index(drop=True).copy()
    if shuffled:
        rng=np.random.default_rng(195)
        for _,ids in source.groupby('k',sort=True).groups.items():
            source.loc[ids,'bias']=source.loc[rng.permutation(ids),'bias'].to_numpy()
    out=entries.copy(); out['correction']=0.
    for k,cur in out.groupby('k',sort=True):
        past=source[source.k.lt(k)]
        weight=.5**((k-past.k)/20)
        numerator=(past.bias*weight).groupby(past.team).sum()
        denominator=weight.groupby(past.team).sum()
        level=numerator/(8+denominator)
        correction=cur.team.map(level).fillna(0.)
        out.loc[cur.index,'correction']=correction-correction.mean()
    return out


def main():
    d=pd.read_csv(ROOT/'results/e194_forecasts_errors_scales.csv',float_precision='round_trip')
    assert d.season.max()==2022
    entries=d[['season','round','k','driver','team']]
    source=d.dropna(subset=['error']).groupby(['k','team']).error.mean().rename('bias').reset_index()
    candidate=bias_predictions(entries,source)
    null=bias_predictions(entries,source,shuffled=True)
    scored=d[['season','round','k','driver','team','y_pace','mean_forecast','error','contender']].copy()
    scored['correction']=candidate.correction
    scored['null_correction']=null.correction
    scored['corrected_mean']=scored.mean_forecast+scored.correction
    scored['base_mse']=scored.error**2
    scored['corrected_mse']=(scored.error-scored.correction)**2
    scored['null_mse']=(scored.error-scored.null_correction)**2
    scored['mse_gain']=scored.base_mse-scored.corrected_mse
    inner=scored[scored.season.ge(2020)]
    assert inner.k.nunique()==61 and len(inner)==1220
    observed=inner[inner.error.notna()]
    primary=observed[observed.contender]
    gain,gain_ci=clustered(primary,['mse_gain'])
    correlation,correlation_ci=clustered(primary,['correction','error'],correlation=True)
    rows=[]
    for (season,round_,k),g in observed.groupby(['season','round','k'],sort=True):
        if len(g)<5:
            continue
        base=g.mean_forecast.corr(g.y_pace,method='spearman')
        corrected=g.corrected_mean.corr(g.y_pace,method='spearman')
        rows.append(dict(season=season,round=round_,k=k,base=base,corrected=corrected,gain=corrected-base))
    ranks=pd.DataFrame(rows)
    rank_gain,rank_ci=clustered(ranks,['gain'])
    reduction=gain/primary.base_mse.mean()
    covered=primary.groupby('k').size().ge(5).sum()
    result=dict(forecast_races=61,forecast_rows=len(inner),label_rows=len(observed),contender_rows=len(primary),covered_races=int(covered),
        primary_mse=primary[['base_mse','corrected_mse','null_mse']].mean().to_dict(),
        mse_gain=gain,mse_gain_ci=gain_ci,relative_mse_reduction=float(reduction),correlation=correlation,correlation_ci=correlation_ci,
        fullfield_mse_gain=float(observed.mse_gain.mean()),pace_rank_races=len(ranks),pace_rank_gain=rank_gain,pace_rank_gain_ci=rank_ci)
    result['gate']=bool(covered>=50 and reduction>=.02 and gain_ci[0]>0 and correlation>=.15 and observed.mse_gain.mean()>=0 and rank_gain>0 and rank_ci[0]>0)
    print('MEASUREMENT',json.dumps(result),flush=True)
    (ROOT/'results/e195_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    scored.to_csv(ROOT/'results/e195_bias_forecasts.csv',index=False)
    source.to_csv(ROOT/'results/e195_team_bias_observations.csv',index=False)
    ranks.to_csv(ROOT/'results/e195_pace_rank_diagnostic.csv',index=False)


if __name__=='__main__':
    main()
