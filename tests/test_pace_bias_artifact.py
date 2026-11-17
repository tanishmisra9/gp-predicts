"""E195 independent scalar history, saved scores and real-source cutoff audit."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from experiments.e194_pace_error_scale import revealed_errors
from experiments.e195_pace_bias import bias_predictions
from tests.test_pace_error_scale_artifact import bootstrap


def scalar(entries, source):
    result=pd.Series(0.,index=entries.index)
    for k,g in entries.groupby('k',sort=True):
        sums={}; weights={}
        for row in source[source.k.lt(k)].itertuples():
            weight=2**(-(k-row.k)/20)
            sums[row.team]=sums.get(row.team,0.)+row.bias*weight
            weights[row.team]=weights.get(row.team,0.)+weight
        values=[sums.get(team,0.)/(8+weights.get(team,0.)) for team in g.team]
        center=sum(values)/len(values)
        result.loc[g.index]=[value-center for value in values]
    return result


def test_artifacts():
    d=pd.read_csv('results/e195_bias_forecasts.csv',float_precision='round_trip')
    original=pd.read_csv('results/e194_forecasts_errors_scales.csv',float_precision='round_trip')
    source=pd.read_csv('results/e195_team_bias_observations.csv',float_precision='round_trip')
    ranks=pd.read_csv('results/e195_pace_rank_diagnostic.csv',float_precision='round_trip')
    summary=json.loads(Path('results/e195_summary.json').read_text())
    assert not d.duplicated(['season','round','driver']).any() and d.season.max()==2022
    for name in ('season','round','k','driver','team','y_pace','mean_forecast','error','contender'):
        pd.testing.assert_series_equal(d[name],original[name],check_exact=True)
    raw=revealed_errors(original)
    np.testing.assert_allclose(raw.error,d.error,equal_nan=True,atol=1e-12)
    expected=raw.dropna(subset=['error']).groupby(['k','team']).error.mean()
    np.testing.assert_allclose(source.set_index(['k','team']).bias,expected,atol=1e-12)
    np.testing.assert_allclose(d.correction,scalar(d,source),atol=1e-12)
    shuffled=source.sort_values(['k','team']).reset_index(drop=True).copy()
    rng=np.random.default_rng(195)
    for _,ids in shuffled.groupby('k',sort=True).groups.items():
        shuffled.loc[ids,'bias']=shuffled.loc[rng.permutation(ids),'bias'].to_numpy()
    np.testing.assert_allclose(d.null_correction,scalar(d,shuffled),atol=1e-12)
    np.testing.assert_allclose(d.groupby('k').correction.mean(),0.,atol=1e-12)
    np.testing.assert_allclose(d.corrected_mean,d.mean_forecast+d.correction,atol=1e-12)
    for name,correction in (('base',0.),('corrected',d.correction),('null',d.null_correction)):
        np.testing.assert_allclose(d[name+'_mse'],(d.error-correction)**2,equal_nan=True,atol=1e-12)
    np.testing.assert_allclose(d.mse_gain,d.base_mse-d.corrected_mse,equal_nan=True,atol=1e-12)
    inner=d[d.season.ge(2020)]; valid=inner.dropna(subset=['error']); primary=valid[valid.contender]
    assert inner.k.nunique()==61 and len(inner)==1220 and len(valid)==1134 and len(primary)==340
    assert primary.groupby('k').size().ge(5).sum()==58
    np.testing.assert_allclose(primary[['base_mse','corrected_mse','null_mse']].mean(),list(summary['primary_mse'].values()),atol=1e-12)
    assert np.isclose(primary.mse_gain.mean(),summary['mse_gain'])
    assert np.isclose(primary.mse_gain.mean()/primary.base_mse.mean(),summary['relative_mse_reduction'])
    assert np.isclose(valid.mse_gain.mean(),summary['fullfield_mse_gain'])
    assert np.isclose(np.corrcoef(primary.correction,primary.error)[0,1],summary['correlation'])
    np.testing.assert_allclose(bootstrap(primary,'mse_gain'),summary['mse_gain_ci'],atol=1e-12)
    np.testing.assert_allclose(bootstrap(primary,'correction','error'),summary['correlation_ci'],atol=1e-12)
    for row in ranks.itertuples():
        g=valid[valid.k.eq(row.k)]
        actual=rankdata(g.y_pace)
        base=np.corrcoef(rankdata(g.mean_forecast),actual)[0,1]
        corrected=np.corrcoef(rankdata(g.corrected_mean),actual)[0,1]
        np.testing.assert_allclose([row.base,row.corrected,row.gain],[base,corrected,corrected-base],atol=1e-12)
    assert ranks.k.nunique()==valid.groupby('k').size().ge(5).sum()
    assert np.isclose(ranks.gain.mean(),summary['pace_rank_gain'])
    np.testing.assert_allclose(bootstrap(ranks,'gain'),summary['pace_rank_gain_ci'],atol=1e-12)
    expected_gate=(summary['covered_races']>=50 and summary['relative_mse_reduction']>=.02 and summary['mse_gain_ci'][0]>0 and summary['correlation']>=.15 and summary['fullfield_mse_gain']>=0 and summary['pace_rank_gain']>0 and summary['pace_rank_gain_ci'][0]>0)
    assert summary['gate']==expected_gate
    target=inner.loc[inner.season.eq(2022)&inner['round'].eq(10),'k'].iloc[0]
    altered=original.copy(); mask=altered.k.ge(target)&altered.y_pace.notna()
    altered.loc[mask,'y_pace']=np.random.default_rng(19).uniform(0,20,mask.sum())
    changed=revealed_errors(altered).dropna(subset=['error']).groupby(['k','team']).error.mean().rename('bias').reset_index()
    entries=d[['season','round','k','driver','team']]
    before=bias_predictions(entries,source); after=bias_predictions(entries,changed)
    pd.testing.assert_frame_equal(before[before.k.le(target)],after[after.k.le(target)],check_exact=True)
    changed.loc[changed.k.lt(target).idxmax(),'bias']+=100.
    prior=bias_predictions(entries,changed)
    assert not before.loc[before.k.eq(target),'correction'].equals(prior.loc[prior.k.eq(target),'correction'])


if __name__=='__main__':
    test_artifacts()
    print('E195 scalar source/history, null, errors, coverage, ranks, bootstrap and real cutoff reproduce')
