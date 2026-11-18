"""E217: structured historical warning-flag observability, no message text."""
import json
import os
from unittest.mock import patch

import fastf1
import pandas as pd
import requests

from experiments.e190_traffic_feasibility import PILOT,cached_session
from experiments.e196_weather_source import restricted_send
from f1.data import ROOT

FIELDS=['Time','Category','Flag','Scope','RacingNumber','Lap']


def allowed_urls(path):
    assert path.startswith('/static/2019/') and path.endswith('_Race/')
    return {host+path+'RaceControlMessages.jsonStream' for host in
        ('https://livetiming.formula1.com','https://livetiming-mirror.fastf1.dev')}


def structured(feed):
    # Never access the free-text Message channel.
    frame=pd.DataFrame({col:feed[col] for col in FIELDS})
    frame['Time']=pd.to_datetime(frame.Time,utc=True)
    return frame


def observe(frame,roster):
    assert set(FIELDS).issubset(frame.columns)
    assert not roster.RacingNumber.duplicated().any() and not roster.driver.duplicated().any()
    distinct=frame[FIELDS].drop_duplicates().copy()
    driver_flags=distinct[distinct.Category.eq('Flag')&distinct.Scope.eq('Driver')]
    warnings=distinct[distinct.Flag.eq('BLACK AND WHITE')].copy()
    good=warnings.Category.eq('Flag')&warnings.Scope.eq('Driver')&warnings.Time.notna()&warnings.RacingNumber.isin(roster.RacingNumber)
    observed=warnings[good].merge(roster,on='RacingNumber',validate='many_to_one')
    counts=distinct[distinct.Flag.notna()].groupby(['Category','Flag','Scope'],dropna=False).size().reset_index(name='count')
    report=dict(rows=len(frame),distinct_rows=len(distinct),flag_rows=int(distinct.Category.eq('Flag').sum()),
        driver_flag_rows=len(driver_flags),warning_rows=len(warnings),assigned_warnings=len(observed),
        unassigned_warnings=int((~good).sum()),flag_schema=counts.astype(object).where(pd.notna(counts),None).to_dict('records'))
    return observed,report


def main():
    import f1.ff1
    from fastf1.internals.f1account import get_token
    if not all(os.environ.get(key) for key in ('F1_USERNAME','F1_PASSWORD')):
        raise RuntimeError('Missing existing F1 credentials')
    fastf1.Cache.offline_mode(False)
    try:
        get_token(method='password')
    except Exception as exc:
        result=dict(sessions=[],gate=False,error_type=type(exc).__name__)
        (ROOT/'results/e217_warning_summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True); return
    reports=[]; parts=[]; warning_parts=[]; rosters=[]
    for season,round_ in PILOT:
        fastf1.Cache.offline_mode(True)
        session=cached_session(fastf1,season,round_); laps=pd.DataFrame(session.laps)
        roster=laps[['DriverNumber','Driver']].dropna().drop_duplicates().rename(columns={'DriverNumber':'RacingNumber','Driver':'driver'})
        roster.RacingNumber=roster.RacingNumber.astype(str)
        rosters.append(roster.assign(season=season,round=round_))
        report=dict(season=season,round=round_,available=False,requests=[])
        fastf1.Cache.offline_mode(False)
        try:
            with patch('requests.adapters.HTTPAdapter.send',new=restricted_send(requests.adapters.HTTPAdapter.send,allowed_urls(session.api_path),report['requests'])):
                feed=fastf1._api.race_control_messages(session.api_path)
        except Exception as exc:
            report['error_type']=type(exc).__name__
        else:
            frame=structured(feed); warnings,counts=observe(frame,roster)
            parts.append(frame.assign(season=season,round=round_))
            warning_parts.append(warnings.assign(season=season,round=round_))
            report.update(available=True,**counts)
        reports.append(report)
        (ROOT/'results/e217_warning_progress.json').write_text(json.dumps(reports,indent=2)+'\n')
        print('SOURCE',json.dumps(report),flush=True)
    good=sum(r.get('available',False) and r.get('rows',0)>=10 and r.get('flag_rows',0)>=5 and r.get('driver_flag_rows',0)>=1 for r in reports)
    supported=sum(r.get('assigned_warnings',0)>=1 for r in reports)
    invalid=sum(r.get('unassigned_warnings',0) for r in reports)
    result=dict(sessions=reports,gate=bool(good==3 and supported>=2 and invalid==0))
    pd.concat(parts,ignore_index=True).to_csv(ROOT/'results/e217_structured_flags.csv',index=False) if parts else None
    pd.concat(warning_parts,ignore_index=True).to_csv(ROOT/'results/e217_warning_rows.csv',index=False) if warning_parts else None
    pd.concat(rosters,ignore_index=True).to_csv(ROOT/'results/e217_driver_numbers.csv',index=False)
    (ROOT/'results/e217_warning_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
