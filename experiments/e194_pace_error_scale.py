"""E194: predict relative-pace error scale; no winner selector or simulation."""
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import f1.walkforward as W
from f1.data import DATA,ROOT,load_results
from f1.features import build_features,pre_weekend_cols


def pace_mean(train,current,F):
    W.assert_pre_weekend(F)
    assert train.k.max()<current.k.min()
    observed=train[train.y_pace.notna()]
    model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=500))
    model.fit(observed[F],observed.y_pace,ridge__sample_weight=W._w(observed,100))
    return model.predict(current[F])


def revealed_errors(frame):
    out=frame.copy(); out['error']=np.nan
    for _,g in out.groupby('k',sort=True):
        valid=g.y_pace.notna()
        if valid.sum()>=5:
            delta=g.loc[valid,'y_pace']-g.loc[valid,'mean_forecast']
            out.loc[delta.index,'error']=delta-delta.mean()
    out['squared_error']=out.error**2
    return out


def forecast_scales(entries,observations,shuffled=False):
    obs=observations.copy().sort_values(['k','team']).reset_index(drop=True)
    if shuffled:
        rng=np.random.default_rng(194)
        for _,ids in obs.groupby('k',sort=True).groups.items():
            obs.loc[ids,'team_mse']=obs.loc[rng.permutation(ids),'team_mse'].to_numpy()
    out=entries.copy(); out['global_scale']=1.; out['team_scale']=1.
    for k,cur in out.groupby('k',sort=True):
        past=obs[obs.k.lt(k)&obs.team_mse.notna()]
        if past.empty:
            continue
        w=.5**((k-past.k)/20)
        global_value=float((past.team_mse*w).sum()/w.sum())
        numerator=(past.team_mse*w).groupby(past.team).sum()
        denominator=w.groupby(past.team).sum()
        team=(numerator+8*global_value)/(denominator+8)
        out.loc[cur.index,'global_scale']=max(global_value,1e-4)
        out.loc[cur.index,'team_scale']=cur.team.map(team).fillna(global_value).clip(lower=1e-4)
    return out


def clustered(frame,columns,correlation=False):
    if frame.empty:
        return np.nan,[np.nan,np.nan]
    groups=[g for _,g in frame.groupby('k',sort=True)]
    def value(g):
        if correlation:
            a=g[columns[0]]; b=g[columns[1]]
            return float(a.corr(b)) if a.std()>0 and b.std()>0 else np.nan
        return float(g[columns[0]].mean())
    rng=np.random.default_rng(194)
    sample=[value(pd.concat([groups[i] for i in rng.integers(len(groups),size=len(groups))])) for _ in range(4000)]
    finite=np.asarray(sample)[np.isfinite(sample)]
    return value(frame),np.percentile(finite,[2.5,97.5]).tolist() if len(finite) else [np.nan,np.nan]


def main():
    results=load_results().loc[lambda d:d.season.le(2022)]
    pace=pd.read_csv(DATA/'racepace_ff1.csv').loc[lambda d:d.season.le(2022)]
    X=build_features(results,pace); F=pre_weekend_cols(X)
    X=X.merge(pace[['season','round','code','pace_gap']].rename(columns={'pace_gap':'y_pace'}),on=['season','round','code'],how='left',validate='one_to_one')
    forecasts=[]
    for _,cur in X[X.season.ge(2019)].groupby('k',sort=True):
        train=X[X.k.lt(cur.k.iloc[0])&X.season.ge(2019)]
        if train.k.nunique()<5:
            continue
        predicted=cur[['season','round','k','driver','team','y_pace','d_season_pts','d10_order']].copy()
        predicted['mean_forecast']=pace_mean(train,cur,F)
        predicted['contender']=False
        score=-cur.d_season_pts.fillna(0)*100+cur.d10_order.fillna(11)*1e-3
        best=cur.assign(selection_score=score).sort_values(['selection_score','driver']).index[:6]
        predicted.loc[best,'contender']=True
        forecasts.append(predicted)
    raw=pd.concat(forecasts,ignore_index=True)
    errors=revealed_errors(raw)
    observations=errors.dropna(subset=['squared_error']).groupby(['k','team'],sort=True).squared_error.mean().rename('team_mse').reset_index()
    predictions=forecast_scales(errors[['season','round','k','driver','team']],observations)
    null=forecast_scales(errors[['season','round','k','driver','team']],observations,shuffled=True)
    scored=errors.copy()
    scored[['global_scale','team_scale']]=predictions[['global_scale','team_scale']]
    scored['null_scale']=null.team_scale
    for name in ('global','team','null'):
        v=scored[name+'_scale']
        scored[name+'_loss']=.5*(np.log(v)+scored.squared_error/v)
        scored[name+'_mse']=(v-scored.squared_error)**2
    scored['loss_gain']=scored.global_loss-scored.team_loss
    scored['mse_gain']=scored.global_mse-scored.team_mse
    inner=scored[scored.season.ge(2020)]
    assert inner.k.nunique()==61 and inner.season.max()==2022
    observed=inner[inner.squared_error.notna()]
    primary=observed[observed.contender]
    loggain,logci=clustered(primary,['loss_gain'])
    corr,corci=clustered(primary,['team_scale','squared_error'],correlation=True)
    qualifying_races=primary.groupby('k').size().ge(5).sum()
    summary=dict(forecast_races=61,forecast_rows=len(inner),label_rows=len(observed),contender_label_rows=len(primary),
        contender_covered_races=int(qualifying_races),logscore_gain=loggain,logscore_gain_ci=logci,
        scale_mse_gain=float(primary.mse_gain.mean()),correlation=corr,correlation_ci=corci,
        fullfield_logscore_gain=float(observed.loss_gain.mean()),
        mean_logscore=primary[['global_loss','team_loss','null_loss']].mean().to_dict())
    summary['gate']=bool(qualifying_races>=50 and loggain>=.01 and logci[0]>0 and primary.mse_gain.mean()>0 and corr>=.15 and observed.loss_gain.mean()>=0)
    print('MEASUREMENT',json.dumps(summary),flush=True)
    (ROOT/'results/e194_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    scored.to_csv(ROOT/'results/e194_forecasts_errors_scales.csv',index=False)
    observations.to_csv(ROOT/'results/e194_team_error_observations.csv',index=False)


if __name__=='__main__':
    main()
