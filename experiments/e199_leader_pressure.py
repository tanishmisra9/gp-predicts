"""E199 cached leader-pressure timing-stream source; no pace response scoring."""
import json
import re
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.e190_traffic_feasibility import cached_session,start_exposure
from f1.data import ROOT,load_results

EXPOSURE=['driver','lap','start','chaser','gap','observed','age','state','stint','compound']


def numeric_gap(value):
    text=str(value).strip()
    value=float(text) if re.fullmatch(r'\+?\d+(?:\.\d+)?',text) else np.nan
    return value if np.isfinite(value) else np.nan


def prepare_stream(stream,identities,field='GapToLeader'):
    assert not identities.DriverNumber.duplicated().any() and not identities.Driver.duplicated().any()
    source=stream.copy(); source['number']=source.Driver.astype(str)
    mapping=identities.assign(DriverNumber=identities.DriverNumber.astype(str)).set_index('DriverNumber').Driver
    source['Driver']=source.number.map(mapping)
    assert source.Driver.notna().all() and source.Time.notna().all()
    source=source.drop_duplicates(['Driver','Time'],keep='last').sort_values(['Driver','Time'],kind='stable').reset_index(drop=True)
    source['gap']=source[field].map(numeric_gap)
    source['observed']=pd.Series(pd.NaT,index=source.index,dtype='timedelta64[ns]')
    for _,g in source.groupby('Driver',sort=True):
        previous=None; observed=pd.NaT
        for index,row in g.iterrows():
            if np.isfinite(row.gap):
                if previous is None or row.gap!=previous:
                    observed=row.Time
                previous=row.gap
            else:
                previous=None; observed=pd.NaT
            source.loc[index,'observed']=observed
    return source


def exposures(source,targets):
    rows=[]
    for target in targets.itertuples():
        snapshot=source[source.Time.le(target.LapStartTime)].groupby('Driver',sort=True).tail(1)
        first=snapshot[snapshot.Position.eq(1)]; second=snapshot[snapshot.Position.eq(2)]
        if len(first)!=1 or len(second)!=1 or first.Driver.iloc[0]!=target.Driver:
            continue
        chaser=second.iloc[0]
        age=(target.LapStartTime-chaser.observed).total_seconds() if pd.notna(chaser.observed) else np.nan
        if not np.isfinite(chaser.gap) or not 0<chaser.gap<=30 or not 0<=age<=30:
            continue
        state='pressured' if chaser.gap<=2 else ('comfortable' if chaser.gap>=5 else 'intermediate')
        rows.append(dict(driver=target.Driver,lap=target.LapNumber,start=target.LapStartTime,chaser=chaser.Driver,
            gap=chaser.gap,observed=chaser.observed,age=age,state=state,stint=target.Stint,compound=target.Compound))
    return pd.DataFrame(rows,columns=EXPOSURE)


def contrasting_blocks(frame):
    result=[]
    for (season,round_,driver,stint),g in frame.groupby(['season','round','driver','stint'],sort=True):
        supported=False
        for _,part in g.dropna(subset=['compound']).groupby(['chaser','compound'],sort=True):
            if part.state.eq('pressured').sum()>=5 and part.state.eq('comfortable').sum()>=5:
                supported=True
        if supported:
            assert g.team.nunique()==1
            result.append(dict(season=int(season),round=int(round_),driver=driver,stint=stint,team=g.team.iloc[0]))
    return pd.DataFrame(result,columns=['season','round','driver','stint','team'])


def main():
    import fastf1
    import f1.ff1
    from fastf1 import _api
    from fastf1.internals.f1account import LivetimingAuthAdapter
    fastf1.Cache.offline_mode(True)
    entries=load_results().loc[lambda d:d.season.eq(2019),['season','round','code','team']]
    assert entries['round'].nunique()==21
    reports=[]; all_exposures=[]
    for round_,event in entries.groupby('round',sort=True):
        report=dict(season=2019,round=int(round_),available=False)
        try:
            session=cached_session(fastf1,2019,int(round_))
            raw=pd.DataFrame(session.laps)
            with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('HTTP forbidden in cached stream pilot')), patch.object(LivetimingAuthAdapter,'_get_token',return_value=None):
                _,stream,_=_api._extended_timing_data(session.api_path)
            identities=raw[['DriverNumber','Driver']].drop_duplicates()
            source=prepare_stream(stream,identities)
            clean=start_exposure(raw); targets=clean[clean.Position.eq(1)&clean.LapNumber.ge(4)]
            current=exposures(source,targets).merge(event[['code','team']].rename(columns={'code':'driver'}),on='driver',how='left',validate='many_to_one')
            assert current.team.notna().all()
            report.update(available=True,stream_rows=len(stream),duplicate_stream_keys=int(stream.duplicated(['Driver','Time']).sum()),
                unknown_gap_rows=int(source.gap.isna().sum()),leader_targets=len(targets),exposures=len(current),
                coverage=float(len(current)/len(targets)) if len(targets) else 0.,states=current.state.value_counts().to_dict())
            all_exposures.append(current.assign(season=2019,round=int(round_)))
        except Exception as exc:
            report['error_type']=type(exc).__name__
        reports.append(report); print(json.dumps(report),flush=True)
    table=pd.concat(all_exposures,ignore_index=True) if all_exposures else pd.DataFrame(columns=EXPOSURE+['team','season','round'])
    blocks=contrasting_blocks(table)
    by_team=blocks.groupby('team').size().astype(int).to_dict()
    covered=sum(r.get('coverage',0)>=.8 for r in reports)
    gate=covered>=15 and len(blocks)>=20 and blocks['round'].nunique()>=8 and sum(n>=3 for n in by_team.values())>=3
    result=dict(sessions=reports,covered_races=covered,exposures=len(table),contrasting_blocks=len(blocks),block_races=int(blocks['round'].nunique()),blocks_by_team=by_team,gate=bool(gate))
    (ROOT/'results/e199_leader_pressure_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    table.to_csv(ROOT/'results/e199_leader_pressure_exposures.csv',index=False)
    blocks.to_csv(ROOT/'results/e199_leader_pressure_blocks.csv',index=False)
    print('SOURCE GATE',bool(gate),flush=True)


if __name__=='__main__':
    main()
