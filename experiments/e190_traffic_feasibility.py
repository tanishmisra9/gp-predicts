"""E190 cache-only feasibility: start-of-lap close-following, no forecast model."""
import json

import numpy as np
import pandas as pd

PILOT=[(2019,5),(2019,12),(2019,14)]
REQUIRED=['Driver','Time','LapStartTime','LapTime','LapNumber','Stint','Compound','TyreLife','Position','PitInTime','PitOutTime','TrackStatus','IsAccurate']


def start_exposure(laps):
    clean=laps[(laps.LapNumber>1)&laps.LapTime.notna()&laps.Time.notna()&laps.LapStartTime.notna()
        &laps.PitInTime.isna()&laps.PitOutTime.isna()&laps.TrackStatus.astype(str).eq('1')
        &laps.IsAccurate.fillna(False)].copy()
    clean['seconds']=clean.LapTime.dt.total_seconds()
    clean=clean[clean.seconds<1.07*clean.groupby('Driver').seconds.transform('median')]
    clean=clean.sort_values(['Time','Driver']).reset_index(drop=True)
    times=clean.Time.dt.total_seconds().to_numpy()
    drivers=clean.Driver.to_numpy()
    gap=[]
    for row in clean.itertuples():
        start=row.LapStartTime.total_seconds()
        lo=np.searchsorted(times,start-1.5,side='left')
        hi=np.searchsorted(times,start,side='right')
        preceding=times[lo:hi][drivers[lo:hi]!=row.Driver]
        gap.append(start-preceding[-1] if len(preceding) else np.nan)
    clean['start_gap']=gap
    clean['close']=clean.start_gap.notna()
    return clean


def disjoint_pairs(clean):
    rows=[]
    for driver,g in clean.groupby('Driver',sort=True):
        g=g.sort_values('LapNumber'); consumed=set()
        for a,b in zip(g.iloc[:-1].itertuples(),g.iloc[1:].itertuples()):
            if a.LapNumber in consumed or b.LapNumber in consumed:
                continue
            valid=b.LapNumber==a.LapNumber+1 and a.Stint==b.Stint and a.Compound==b.Compound
            valid=valid and np.isclose(b.TyreLife-a.TyreLife,1.) and a.Position==b.Position
            if valid and a.close!=b.close:
                rows.append(dict(driver=driver,lap1=a.LapNumber,lap2=b.LapNumber,entering=bool(b.close),stint=b.Stint))
                consumed.update((a.LapNumber,b.LapNumber))
    return pd.DataFrame(rows,columns=['driver','lap1','lap2','entering','stint'])


def cached_session(fastf1, season, round_):
    from unittest.mock import patch
    with patch('requests.adapters.HTTPAdapter.send', side_effect=RuntimeError('HTTP forbidden in cache-only pilot')):
        session=fastf1.get_session(season,round_,'R')
        session.load(laps=True,telemetry=False,weather=False,messages=False)
    return session


def main():
    import fastf1
    import f1.ff1  # existing project cache initialization
    fastf1.Cache.offline_mode(True)
    reports=[]; all_pairs=[]
    for season,round_ in PILOT:
        report=dict(season=season,round=round_)
        try:
            session=cached_session(fastf1,season,round_)
        except Exception as exc:
            report.update(available=False,error_type=type(exc).__name__)
            reports.append(report)
            print(json.dumps(report),flush=True)
            continue
        missing=sorted(set(REQUIRED)-set(session.laps.columns))
        report['missing_fields']=missing
        if missing:
            report['available']=False
        else:
            raw=pd.DataFrame(session.laps)
            clean=start_exposure(raw); pairs=disjoint_pairs(clean)
            entrants=sorted(raw.Driver.dropna().unique())
            counts=pairs.groupby('driver').size().reindex(entrants,fill_value=0)
            report.update(available=True,raw_laps=len(raw),clean_laps=len(clean),drivers=len(entrants),pairs=len(pairs),
                drivers_with_two_pairs=int(counts.ge(2).sum()),by_driver=counts.astype(int).to_dict(),
                entering=int(pairs.entering.sum()),leaving=int((~pairs.entering).sum()))
            all_pairs.append(pairs.assign(season=season,round=round_))
        reports.append(report)
        print(json.dumps(report),flush=True)
    gate=all(r.get('available',False) for r in reports) and sum(r.get('pairs',0) for r in reports)>=60
    gate=gate and sum(r.get('pairs',0)>=20 for r in reports)>=2
    gate=gate and sum(r.get('drivers_with_two_pairs',0)>=r.get('drivers',1)/2 for r in reports if r.get('available'))>=2
    print('FEASIBILITY GATE',bool(gate),flush=True)
    from f1.data import ROOT
    (ROOT/'results/e190_cache_feasibility.json').write_text(json.dumps(dict(sessions=reports,gate=bool(gate)),indent=2)+'\n')
    pd.concat(all_pairs,ignore_index=True).to_csv(ROOT/'results/e190_traffic_pairs.csv',index=False) if all_pairs else None


if __name__=='__main__':
    main()
