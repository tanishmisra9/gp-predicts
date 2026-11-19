"""E234 fixed 2019 driver-launch temporal transfer; no winner fit."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.e190_traffic_feasibility import cached_session
from experiments.e191_traffic_repeatability import past_response,cluster_interval
from experiments.e233_launch_source import event
from f1.data import ROOT


def history(source):
    assert source.season.eq(2019).all()
    assert not source.duplicated(['round','driver']).any()
    table=source.rename(columns={'driver':'team','response':'centered_response'})
    return past_response(table).rename(columns={'team':'driver','centered_response':'response'})


def evaluate(source,source_gate):
    forecast=history(source)
    eligible=forecast[forecast.past_count.ge(3)&forecast.response.notna()].copy()
    result=dict(source_gate=bool(source_gate),temporal_rows=len(eligible),temporal_races=int(eligible['round'].nunique()),gate=False)
    result['coverage_gate']=bool(source_gate and len(eligible)>=120 and eligible['round'].nunique()>=12)
    if not result['coverage_gate']:
        return result,forecast
    corr,ci=cluster_interval(eligible,'prediction','response')
    eligible['mse_gain']=eligible.response**2-(eligible.prediction-eligible.response)**2
    gain,gain_ci=cluster_interval(eligible,'mse_gain')
    null=source.copy(); rng=np.random.default_rng(234)
    for _,ids in null[null.response.notna()].groupby('round').groups.items():
        null.loc[ids,'response']=null.loc[rng.permutation(ids),'response'].to_numpy()
    shuffled=history(null)
    eligible['null_gain']=(shuffled.prediction.reindex(eligible.index)-eligible.response)**2-(eligible.prediction-eligible.response)**2
    ng,nci=cluster_interval(eligible,'null_gain')
    result.update(correlation=corr,correlation_ci=ci.tolist(),baseline_mse=float((eligible.response**2).mean()),
        forecast_mse=float(((eligible.prediction-eligible.response)**2).mean()),mse_gain=gain,mse_gain_ci=gain_ci.tolist(),
        null_mse_gain=ng,null_mse_gain_ci=nci.tolist())
    result['gate']=bool(corr>=.15 and ci[0]>0 and gain>0 and gain_ci[0]>0 and ng>0 and nci[0]>0)
    return result,forecast


def brackets(car,origin,t0):
    c=car[car.Date.notna()].drop_duplicates('Date',keep='last').sort_values('Date')
    t=(c.Date-origin).dt.total_seconds().to_numpy(); v=c.Speed.to_numpy(float)
    i=int(np.flatnonzero((v[:-1]<=20)&(v[1:]>20)&(t[1:]>=t0)&(t[:-1]<=t0+10))[0])
    return float(t[i]),float(t[i+1])


def main():
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True); reports=[]; rows=[]; widths=[]; robust=0; pairs=0
    for round_ in range(1,22):
        session=cached_session(fastf1,2019,round_)
        with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('launch transfer HTTP forbidden')):
            cars=fastf1._api.car_data(session.api_path)
        report=event(pd.DataFrame(session.laps),cars)
        origin=max((c.Date-c.Time).max() for c in cars.values() if c.Time.notna().any()).round('ms')
        for r in report['rows']:
            if r['supported']:
                r['bracket20_low'],r['bracket20_high']=brackets(cars[r['number']],origin,report['t0'])
        good=[r for r in report['rows'] if r['supported']] if report['gate'] else []
        for i,a in enumerate(good):
            widths.append(a['bracket20_high']-a['bracket20_low'])
            for b in good[i+1:]:
                pairs+=1
                robust+=a['bracket20_high']<b['bracket20_low'] or b['bracket20_high']<a['bracket20_low']
        for r in report['rows']:
            rows.append(dict(season=2019,round=round_,driver=r['driver'],response=r.get('relative20',np.nan) if report['gate'] else np.nan))
        reports.append(dict(season=2019,round=round_,**report))
        print('SOURCE',round_,report['supported'],report['gate'],flush=True)
    source=pd.DataFrame(rows)
    count=sum(r['gate'] for r in reports); measured=int(source.response.notna().sum())
    source_gate=bool(count>=16 and measured>=300)
    result,forecast=evaluate(source,source_gate)
    result.update(valid_events=count,measured_rows=measured,mean_bracket_width=float(np.mean(widths)) if widths else None,
        robust_pairs=int(robust),pairs=pairs,robust_pair_fraction=robust/pairs if pairs else None,winner_fit=False)
    (ROOT/'results/e234_launch_source.json').write_text(json.dumps(dict(sessions=reports,gate=source_gate,winner_fit=False,reaction_time_verified=False),indent=2)+'\n')
    source.to_csv(ROOT/'results/e234_driver_measurements.csv',index=False)
    forecast.to_csv(ROOT/'results/e234_temporal_forecasts.csv',index=False)
    (ROOT/'results/e234_transfer_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
