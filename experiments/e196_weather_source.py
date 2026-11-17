"""E196: bounded historical minute-weather acquisition; no forecast scoring."""
import json
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

from experiments.e190_traffic_feasibility import PILOT,cached_session,start_exposure
from f1.data import ROOT


def align_weather(laps,weather):
    source=weather[['Time','TrackTemp','Rainfall']].rename(columns={'Time':'weather_time'}).copy()
    assert not source.weather_time.duplicated().any()
    return pd.merge_asof(laps.sort_values('LapStartTime'),source.sort_values('weather_time'),
        left_on='LapStartTime',right_on='weather_time',direction='backward',tolerance=pd.Timedelta(seconds=120))


def allowed_urls(path):
    assert path.startswith('/static/2019/') and path.endswith('/')
    return {host+path+'WeatherData.jsonStream' for host in ('https://livetiming.formula1.com','https://livetiming-mirror.fastf1.dev')}


def restricted_send(send,allowed,audit=None):
    def wrapped(adapter,request,**kwargs):
        if request.url not in allowed:
            raise RuntimeError('Non-pilot-weather network request forbidden')
        try:
            response=send(adapter,request,**kwargs)
        except Exception as exc:
            if audit is not None:
                audit.append(dict(url=request.url,error_type=type(exc).__name__))
            raise
        if audit is not None:
            audit.append(dict(url=request.url,status=int(response.status_code)))
        return response
    return wrapped


def main(experiment=196,authenticated=False):
    import fastf1
    from fastf1.internals.f1account import LivetimingAuthAdapter
    import f1.ff1  # existing project cache setup only
    reports=[]; samples=[]; aligned=[]
    for season,round_ in PILOT:
        report=dict(season=season,round=round_,available=False,requests=[])
        try:
            assert season==2019
            fastf1.Cache.offline_mode(True)
            session=cached_session(fastf1,season,round_)
            clean=start_exposure(pd.DataFrame(session.laps))
            allowed=allowed_urls(session.api_path)
            fastf1.Cache.offline_mode(False)
            send=requests.adapters.HTTPAdapter.send
            # Public mode disables token lookup; account mode uses the normal adapter after pre-auth.
            with patch('requests.adapters.HTTPAdapter.send',new=restricted_send(send,allowed,report['requests'])), (nullcontext() if authenticated else patch.object(LivetimingAuthAdapter,'_get_token',return_value=None)):
                session.load(laps=False,telemetry=False,weather=True,messages=False)
            weather=pd.DataFrame(session.weather_data).copy()
            assert len(weather)>0 and weather.Time.notna().all() and not weather.Time.duplicated().any()
            merged=align_weather(clean,weather)
            matched=merged.weather_time.notna()&merged.TrackTemp.notna()
            dry=matched&merged.Rainfall.eq(False)
            counts=merged[dry].groupby('Driver').size()
            span=float(merged.loc[dry,'TrackTemp'].quantile(.95)-merged.loc[dry,'TrackTemp'].quantile(.05))
            report.update(available=True,weather_rows=len(weather),finite_track_fraction=float(np.isfinite(weather.TrackTemp).mean()),
                clean_laps=len(clean),matched_laps=int(matched.sum()),match_fraction=float(matched.mean()),dry_laps=int(dry.sum()),
                drivers_with_20_dry=int(counts.ge(20).sum()),dry_track_p95_p05=span)
            samples.append(weather.assign(season=season,round=round_))
            keep=['Driver','LapNumber','LapStartTime','seconds','Stint','Compound','TyreLife','Position','close','weather_time','TrackTemp','Rainfall']
            aligned.append(merged[keep].assign(season=season,round=round_))
        except Exception as exc:
            report.update(error_type=type(exc).__name__)
        finally:
            fastf1.Cache.offline_mode(True)
        reports.append(report)
        print(json.dumps(report),flush=True)
    gate=all(r['available'] and r['weather_rows']>=100 and r['finite_track_fraction']>=.95 and r['match_fraction']>=.9 for r in reports)
    gate=gate and sum(r.get('dry_track_p95_p05',0)>=3 and r.get('drivers_with_20_dry',0)>=10 for r in reports)>=2
    result=dict(sessions=reports,gate=bool(gate),authenticated=bool(authenticated))
    (ROOT/f'results/e{experiment}_weather_source_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if samples:
        pd.concat(samples,ignore_index=True).to_csv(ROOT/f'results/e{experiment}_weather_samples.csv',index=False)
    if aligned:
        pd.concat(aligned,ignore_index=True).to_csv(ROOT/f'results/e{experiment}_weather_aligned_laps.csv',index=False)
    print('FEASIBILITY GATE',bool(gate),flush=True)


if __name__=='__main__':
    main()
