"""E210 fixed 2019 stationary pit intervals: repeatability and prior forecast."""
import json
import os
from unittest.mock import patch

import fastf1
import pandas as pd
import requests

from experiments.e190_traffic_feasibility import cached_session
from experiments.e196_weather_source import restricted_send
from experiments.e204_warmup import measurements,evaluate
from experiments.e209_stationary_source import allowed_urls,measure
from f1.data import ROOT,load_results

SOURCE=ROOT/'results/e210_stationary_windows.csv'
STATUS=ROOT/'results/e210_source_status.json'


def team_table(windows,entries):
    assert entries.season.eq(2019).all() and windows.season.eq(2019).all()
    assert not windows.duplicated(['season','round','driver','pitin_lap']).any()
    source=windows[windows.usable].copy()
    source=source.merge(entries.rename(columns={'code':'driver'}),on=['season','round','driver'],how='left',validate='many_to_one')
    assert source.team.notna().all()
    source['contrast']=source.midpoint
    return measurements(source,entries)


def extract():
    import f1.ff1
    from fastf1.internals.f1account import get_token
    reports=json.loads(STATUS.read_text()) if STATUS.exists() else []
    done={r['round'] for r in reports}
    assert len(done)==len(reports) and done.issubset(set(range(1,22)))
    if reports:
        assert all(r['season']==2019 for r in reports) and SOURCE.exists()
    windows=pd.read_csv(SOURCE,float_precision='round_trip') if SOURCE.exists() else pd.DataFrame()
    # A source write interrupted before its status write is recomputed from cache.
    if len(windows):
        windows=windows[windows['round'].isin(done)].copy()
    if len(done)<21:
        if not all(os.environ.get(k) for k in ('F1_USERNAME','F1_PASSWORD')):
            raise RuntimeError('Missing existing F1 credentials')
        fastf1.Cache.offline_mode(False)
        try:
            get_token(method='password')
        except Exception as exc:
            error=dict(error_type=type(exc).__name__,stage='existing_account_authentication')
            (ROOT/'results/e210_acquisition_error.json').write_text(json.dumps(error,indent=2)+'\n')
            raise RuntimeError('Existing account authentication failed') from None
    for round_ in range(1,22):
        if round_ in done:
            continue
        fastf1.Cache.offline_mode(True)
        session=cached_session(fastf1,2019,round_); laps=pd.DataFrame(session.laps)
        report=dict(season=2019,round=round_,available=False,requests=[])
        fastf1.Cache.offline_mode(False)
        try:
            with patch('requests.adapters.HTTPAdapter.send',new=restricted_send(requests.adapters.HTTPAdapter.send,allowed_urls(session.api_path),report['requests'])):
                cars=fastf1._api.car_data(session.api_path)
        except Exception as exc:
            report['error_type']=type(exc).__name__
        else:
            current,origin=measure(laps,cars)
            windows=pd.concat([windows,current.assign(season=2019,round=round_)],ignore_index=True)
            report.update(available=True,pit_windows=len(current),usable=int(current.usable.sum()),
                car_rows=sum(len(c) for c in cars.values()),padded_rows=sum(int(c.Time.isna().sum()) for c in cars.values()),origin=str(origin))
        reports.append(report)
        windows.to_csv(SOURCE,index=False)
        STATUS.write_text(json.dumps(reports,indent=2)+'\n')
        print('SOURCE',json.dumps(report),flush=True)
    return windows,reports


def main():
    windows,reports=extract()
    entries=load_results().loc[lambda d:d.season.eq(2019),['season','round','code','team']]
    assert set(entries['round'])==set(range(1,22))
    table=team_table(windows,entries)
    table.to_csv(ROOT/'results/e210_team_measurements.csv',index=False)
    result,forecast=evaluate(table)
    result.update(source_events=len(reports),available=sum(r['available'] for r in reports),
        pit_windows=len(windows),usable_windows=int(windows.usable.sum()))
    forecast.to_csv(ROOT/'results/e210_temporal_forecasts.csv',index=False)
    (ROOT/'results/e210_measurement_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
