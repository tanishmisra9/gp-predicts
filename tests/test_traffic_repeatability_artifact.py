"""Independent E191 pair aggregation, temporal sums and bootstrap reconstruction."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.e191_traffic_repeatability import past_response,team_measurements


def response(p,min_count,min_each):
    enter=p[p.entering].contrast.to_numpy(); leave=p[~p.entering].contrast.to_numpy()
    if len(p)<min_count or min(len(enter),len(leave))<min_each:
        return np.nan
    return (sum(enter)/len(enter)+sum(leave)/len(leave))/2


def bootstrap_stats(frame,x,y=None):
    q=frame.copy(); q['n']=1.
    if y is None:
        q['total']=q[x]; cols=['n','total']
    else:
        q['sx']=q[x]; q['sy']=q[y]; q['xx']=q[x]**2; q['yy']=q[y]**2; q['xy']=q[x]*q[y]
        cols=['n','sx','sy','xx','yy','xy']
    stats=q.groupby('round',sort=True)[cols].sum().to_numpy()
    choices=np.random.default_rng(191).integers(len(stats),size=(4000,len(stats)))
    weights=np.zeros((4000,len(stats)))
    np.add.at(weights,(np.repeat(np.arange(4000),len(stats)),choices.ravel()),1)
    summed=weights@stats
    if y is None:
        value=summed[:,1]/summed[:,0]
    else:
        n,sx,sy,xx,yy,xy=summed.T
        value=(xy-sx*sy/n)/np.sqrt((xx-sx*sx/n)*(yy-sy*sy/n))
    return np.percentile(value[np.isfinite(value)],[2.5,97.5])


def test_artifacts():
    pairs=pd.read_csv('results/e191_source_pairs.csv',float_precision='round_trip')
    table=pd.read_csv('results/e191_team_measurements.csv',float_precision='round_trip')
    forecast=pd.read_csv('results/e191_temporal_forecasts.csv',float_precision='round_trip')
    summary=json.loads(Path('results/e191_measurement_summary.json').read_text())
    statuses=json.loads(Path('results/e191_source_status.json').read_text())
    assert len(statuses)==21 and all(s['available'] for s in statuses)
    assert pairs.season.eq(2019).all() and len(pairs)==2163==sum(s['pairs'] for s in statuses)
    assert len(table)==210 and table['round'].nunique()==21
    for _,g in pairs.groupby(['round','driver']):
        used=list(g.lap1)+list(g.lap2)
        assert len(used)==len(set(used)) and g.lap2.sub(g.lap1).eq(1).all()
    signed=np.where(pairs.entering,1.,-1.)
    np.testing.assert_allclose(pairs.contrast,(pairs.seconds2-pairs.seconds1)*200/(pairs.seconds1+pairs.seconds2)*signed,atol=1e-12)
    actual=table.copy()
    for i,row in actual.iterrows():
        p=pairs[pairs['round'].eq(row['round'])&pairs.team.eq(row.team)].sort_values(['completion2','driver','lap1'])
        assert len(p)==row.pairs
        expected=[response(p,8,2),response(p.iloc[::2],4,1),response(p.iloc[1::2],4,1)]
        np.testing.assert_allclose(row[['response','half0','half1']].to_numpy(dtype=float),expected,equal_nan=True,atol=1e-12)
    for col in ('response','half0','half1'):
        expected=table[col]-table.groupby('round')[col].transform('mean')
        expected=expected.where(table.groupby('round')[col].transform('count').ge(6))
        np.testing.assert_allclose(table['centered_'+col],expected,equal_nan=True,atol=1e-12)
    for row in forecast.itertuples():
        earlier=table[table['round'].lt(row.round)&table.team.eq(row.team)&table.centered_response.notna()]
        numerator=denominator=0.
        for source in earlier.itertuples():
            weight=.5**((row.round-source.round)/10)
            numerator+=source.centered_response*weight; denominator+=weight
        assert row.past_count==len(earlier)
        assert np.isclose(row.prediction,numerator/(5+denominator),atol=1e-12)
    halves=table.dropna(subset=['centered_half0','centered_half1'])
    eligible=forecast[forecast.past_count.ge(3)&forecast.centered_response.notna()].copy()
    assert len(halves)==112==summary['split_pairs'] and len(eligible)==99==summary['temporal_rows']
    split=np.corrcoef(halves.centered_half0,halves.centered_half1)[0,1]
    temporal=np.corrcoef(eligible.prediction,eligible.centered_response)[0,1]
    assert np.isclose(split,summary['split_correlation']) and np.isclose(temporal,summary['temporal_correlation'])
    eligible['gain']=eligible.centered_response**2-(eligible.prediction-eligible.centered_response)**2
    assert np.isclose(eligible.gain.mean(),summary['mse_gain'])
    np.testing.assert_allclose(bootstrap_stats(halves,'centered_half0','centered_half1'),summary['split_ci'],atol=1e-12)
    np.testing.assert_allclose(bootstrap_stats(eligible,'prediction','centered_response'),summary['temporal_ci'],atol=1e-12)
    np.testing.assert_allclose(bootstrap_stats(eligible,'gain'),summary['mse_gain_ci'],atol=1e-12)
    assert np.isclose((eligible.centered_response**2).mean(),summary['baseline_mse'])
    assert np.isclose(((eligible.prediction-eligible.centered_response)**2).mean(),summary['forecast_mse'])
    shuffled=table.copy(); rng=np.random.default_rng(191)
    for _,ids in shuffled[shuffled.centered_response.notna()].groupby('round').groups.items():
        shuffled.loc[ids,'centered_response']=shuffled.loc[rng.permutation(ids),'centered_response'].to_numpy()
    errors=[]
    for row in eligible.itertuples():
        earlier=shuffled[shuffled['round'].lt(row.round)&shuffled.team.eq(row.team)&shuffled.centered_response.notna()]
        weights=[.5**((row.round-z.round)/10) for z in earlier.itertuples()]
        prediction=sum(w*z for w,z in zip(weights,earlier.centered_response))/(5+sum(weights))
        errors.append((prediction-row.centered_response)**2)
    assert np.isclose(np.mean(errors),summary['null_mse'])
    assert not summary['gate'] and summary['valid_races']==16
    entries=table[['season','round','team']]
    changed=pairs.copy(); changed.loc[changed['round'].ge(20),'contrast']+=100.
    revised=past_response(team_measurements(changed,entries))
    target=forecast['round'].eq(20)
    pd.testing.assert_frame_equal(forecast.loc[target,['prediction','past_count']],revised.loc[target,['prediction','past_count']],check_exact=True)
    source=table[table.centered_response.notna()&table['round'].lt(20)].iloc[0]
    changed.loc[changed['round'].eq(source['round'])&changed.team.eq(source.team),'contrast']+=50.
    earlier=past_response(team_measurements(changed,entries))
    assert not forecast.loc[target,'prediction'].equals(earlier.loc[target,'prediction'])


if __name__=='__main__':
    test_artifacts()
    print('E191 raw pair math, disjoint halves, all temporal sums, bootstrap CIs and real-source cutoff reproduce')
