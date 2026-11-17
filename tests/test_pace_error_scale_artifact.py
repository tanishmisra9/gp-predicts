"""E194 scalar-source, score, coverage and independently bootstrapped audit."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.e194_pace_error_scale import forecast_scales,revealed_errors


def scalar_scales(frame,observations):
    out=frame[['k','team']].copy(); out['global_scale']=1.; out['team_scale']=1.
    for k,g in out.groupby('k',sort=True):
        prior=observations[observations.k.lt(k)]
        total=weight_total=0.; sums={}; weights={}
        for row in prior.itertuples():
            w=.5**((k-row.k)/20)
            total+=w*row.team_mse; weight_total+=w
            sums[row.team]=sums.get(row.team,0.)+w*row.team_mse
            weights[row.team]=weights.get(row.team,0.)+w
        if weight_total:
            global_value=total/weight_total
            out.loc[g.index,'global_scale']=max(global_value,1e-4)
            for i,row in g.iterrows():
                out.loc[i,'team_scale']=max((sums.get(row.team,0.)+8*global_value)/(weights.get(row.team,0.)+8),1e-4)
    return out


def bootstrap(frame,x,y=None):
    q=frame.copy(); q['n']=1.
    if y is None:
        q['total']=q[x]; cols=['n','total']
    else:
        q['sx']=q[x]; q['sy']=q[y]; q['xx']=q[x]**2; q['yy']=q[y]**2; q['xy']=q[x]*q[y]
        cols=['n','sx','sy','xx','yy','xy']
    stats=q.groupby('k',sort=True)[cols].sum().to_numpy()
    ids=np.random.default_rng(194).integers(len(stats),size=(4000,len(stats)))
    weights=np.zeros((4000,len(stats)))
    np.add.at(weights,(np.repeat(np.arange(4000),len(stats)),ids.ravel()),1)
    sums=weights@stats
    if y is None:
        values=sums[:,1]/sums[:,0]
    else:
        n,sx,sy,xx,yy,xy=sums.T
        values=(xy-sx*sy/n)/np.sqrt((xx-sx*sx/n)*(yy-sy*sy/n))
    return np.percentile(values[np.isfinite(values)],[2.5,97.5])


def test_artifacts():
    d=pd.read_csv('results/e194_forecasts_errors_scales.csv',float_precision='round_trip')
    obs=pd.read_csv('results/e194_team_error_observations.csv',float_precision='round_trip')
    result=json.loads(Path('results/e194_summary.json').read_text())
    assert not d.duplicated(['season','round','driver']).any() and d.season.max()==2022
    for _,g in d.groupby('k',sort=True):
        finite=g.y_pace.notna()
        if finite.sum()<5:
            assert g.error.isna().all()
        else:
            delta=(g.loc[finite,'y_pace']-g.loc[finite,'mean_forecast']).to_numpy()
            np.testing.assert_allclose(g.loc[finite,'error'],delta-sum(delta)/len(delta),atol=1e-12)
    np.testing.assert_allclose(d.squared_error,d.error**2,equal_nan=True,atol=1e-12)
    expected=d.dropna(subset=['squared_error']).groupby(['k','team']).squared_error.mean()
    np.testing.assert_allclose(obs.set_index(['k','team']).team_mse,expected,atol=1e-12)
    scalar=scalar_scales(d,obs)
    np.testing.assert_allclose(d[['global_scale','team_scale']],scalar[['global_scale','team_scale']],atol=1e-12)
    null=obs.sort_values(['k','team']).reset_index(drop=True).copy(); rng=np.random.default_rng(194)
    for _,ids in null.groupby('k',sort=True).groups.items():
        null.loc[ids,'team_mse']=null.loc[rng.permutation(ids),'team_mse'].to_numpy()
    null_prediction=scalar_scales(d,null)
    np.testing.assert_allclose(d.null_scale,null_prediction.team_scale,atol=1e-12)
    for name in ('global','team','null'):
        scale=d[name+'_scale']
        np.testing.assert_allclose(d[name+'_loss'],.5*(np.log(scale)+d.squared_error/scale),equal_nan=True,atol=1e-12)
        np.testing.assert_allclose(d[name+'_mse'],(scale-d.squared_error)**2,equal_nan=True,atol=1e-12)
    inner=d[d.season.ge(2020)]; valid=inner[inner.squared_error.notna()]; primary=valid[valid.contender]
    assert inner.k.nunique()==61 and len(inner)==1220 and len(valid)==1134 and len(primary)==340
    assert primary.groupby('k').size().ge(5).sum()==58
    baseline=pd.read_csv('results/e181_inner_predictions.csv',float_precision='round_trip')
    for (s,r),g in inner.groupby(['season','round']):
        old=baseline[baseline.season.eq(s)&baseline['round'].eq(r)&baseline.model.eq('standings')].sort_values(['score','driver']).driver.iloc[:6]
        assert set(g[g.contender].driver)==set(old)
    gain=primary.global_loss-primary.team_loss
    assert np.isclose(gain.mean(),result['logscore_gain'])
    assert np.isclose(primary.mse_gain.mean(),result['scale_mse_gain'])
    assert np.isclose(valid.loss_gain.mean(),result['fullfield_logscore_gain'])
    corr=np.corrcoef(primary.team_scale,primary.squared_error)[0,1]
    assert np.isclose(corr,result['correlation'])
    np.testing.assert_allclose(bootstrap(primary,'loss_gain'),result['logscore_gain_ci'],atol=1e-12)
    np.testing.assert_allclose(bootstrap(primary,'team_scale','squared_error'),result['correlation_ci'],atol=1e-12)
    assert not result['gate']
    target_k=inner.loc[inner.season.eq(2022)&inner['round'].eq(10),'k'].iloc[0]
    altered=d.copy(); mask=altered.k.ge(target_k)&altered.y_pace.notna()
    altered.loc[mask,'y_pace']=np.random.default_rng(7).uniform(0,20,mask.sum())
    changed=revealed_errors(altered).dropna(subset=['squared_error']).groupby(['k','team']).squared_error.mean().rename('team_mse').reset_index()
    prediction=forecast_scales(d[['season','round','k','driver','team']],changed)
    actual=forecast_scales(d[['season','round','k','driver','team']],obs)
    pd.testing.assert_frame_equal(actual.loc[d.k.eq(target_k)],prediction.loc[d.k.eq(target_k)],check_exact=True)
    changed.loc[changed.k.lt(target_k).idxmax(),'team_mse']+=100.
    earlier=forecast_scales(d[['season','round','k','driver','team']],changed)
    assert not actual.loc[d.k.eq(target_k),'global_scale'].equals(earlier.loc[d.k.eq(target_k),'global_scale'])


if __name__=='__main__':
    test_artifacts()
    print('E194 source sums, observed coverage/selection, scales/scores, null, CIs and real cutoff reproduce')
