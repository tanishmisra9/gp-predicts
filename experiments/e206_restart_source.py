"""E206 fixed cached rolling-restart source pilot, no forecast fit."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.e190_traffic_feasibility import cached_session
from f1.data import ROOT

PILOT=[5,10,20]
COLS=['restart','green','before','after','driver','position_before','position_after','rank_before','rank_after','gain','age_before','age_after']


def extract_restarts(laps,status,stream):
    assert not laps.duplicated(['Driver','LapNumber']).any()
    assert status.Time.notna().all() and not status.Time.duplicated().any()
    flags=status.sort_values('Time').copy(); flags['previous']=flags.Status.shift(1)
    starts=flags[flags.Status.eq('1')&flags.previous.eq('4')]
    source=stream.copy()
    source=source.drop_duplicates(['Driver','Time'],keep='last').sort_values('Time',kind='stable')
    assert source.Time.notna().all()
    rows=[]; reports=[]
    for number,item in enumerate(starts.itertuples(),1):
        green=item.Time; before=green-pd.Timedelta(seconds=5); after=green+pd.Timedelta(seconds=120)
        previous=flags[flags.Time.le(before)].tail(1)
        clear=previous.Status.eq('4').all() and len(previous)==1
        clear=clear and flags[flags.Time.gt(green)&flags.Time.le(after)].Status.eq('1').all()
        candidates=[]
        if clear:
            a=source[source.Time.le(before)].groupby('Driver',sort=True).tail(1).set_index('Driver')
            b=source[source.Time.le(after)].groupby('Driver',sort=True).tail(1).set_index('Driver')
            for driver in sorted(set(a.index)&set(b.index)):
                age_a=(before-a.loc[driver,'Time']).total_seconds(); age_b=(after-b.loc[driver,'Time']).total_seconds()
                positions=[a.loc[driver,'Position'],b.loc[driver,'Position']]
                if not 0<=age_a<=30 or not 0<=age_b<=30 or not np.isfinite(positions).all() or not all(1<=v<=30 and float(v).is_integer() for v in positions):
                    continue
                g=laps[laps.Driver.eq(driver)]
                overlap=g[g.Time.ge(before)&g.LapStartTime.le(after)]
                if overlap.empty or overlap.PitInTime.notna().any() or overlap.PitOutTime.notna().any():
                    continue
                full=g[g.LapStartTime.ge(green)&g.Time.le(after)&g.TrackStatus.astype(str).eq('1')&g.IsAccurate.fillna(False)]
                if full.empty or g.Time.max()<after:
                    continue
                candidates.append(dict(restart=number,green=green,before=before,after=after,driver=driver,
                    position_before=positions[0],position_after=positions[1],age_before=age_a,age_after=age_b))
        frame=pd.DataFrame(candidates)
        unique=bool(len(frame)>0 and not frame.position_before.duplicated().any() and not frame.position_after.duplicated().any())
        usable=bool(clear and unique and len(frame)>=6)
        reports.append(dict(restart=number,green_seconds=green.total_seconds(),clear_window=bool(clear),eligible_drivers=len(frame),unique_positions=unique,usable=usable))
        if usable:
            frame['rank_before']=frame.position_before.rank(method='min').astype(int)
            frame['rank_after']=frame.position_after.rank(method='min').astype(int)
            frame['gain']=frame.rank_before-frame.rank_after
            rows.append(frame[COLS])
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(columns=COLS),reports


def main():
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True); parts=[]; reports=[]
    for round_ in PILOT:
        session=cached_session(fastf1,2019,round_)
        laps=pd.DataFrame(session.laps)
        identities=laps[['DriverNumber','Driver']].drop_duplicates()
        assert not identities.DriverNumber.duplicated().any()
        mapping=identities.assign(DriverNumber=identities.DriverNumber.astype(str)).set_index('DriverNumber').Driver
        with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('HTTP forbidden in cached restart pilot')):
            _,stream,_=fastf1._api._extended_timing_data(session.api_path)
        stream=stream.copy(); stream.Driver=stream.Driver.astype(str).map(mapping)
        assert stream.Driver.notna().all()
        measured,events=extract_restarts(laps,pd.DataFrame(session.track_status),stream)
        parts.append(measured.assign(season=2019,round=round_))
        reports.append(dict(season=2019,round=round_,events=events,rows=len(measured)))
    table=pd.concat(parts,ignore_index=True)
    counts=[sum(e['usable'] for e in r['events']) for r in reports]
    result=dict(sessions=reports,opportunities=len(table),usable_restarts=sum(counts),
        gate=bool(sum(c>0 for c in counts)>=2 and sum(counts)>=3 and len(table)>=30))
    table.to_csv(ROOT/'results/e206_restart_source.csv',index=False)
    (ROOT/'results/e206_restart_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
