"""E237 fixed post-SC unchanged-tyre four-lap support, no pace response fit."""
import json
import numpy as np
import pandas as pd

from experiments.e190_traffic_feasibility import cached_session,start_exposure
from f1.data import ROOT,load_results

COLS=['restart','green_seconds','sc_seconds','driver','previous_lap','first_full_lap','stint','compound']


def blocks(laps,status):
    assert not laps.duplicated(['Driver','LapNumber']).any()
    assert status.Time.notna().all() and not status.Time.duplicated().any()
    flags=status.sort_values('Time').copy()
    changes=flags[flags.Status.ne(flags.Status.shift(1))].copy()
    changes['previous']=changes.Status.shift(1)
    changes['duration']=changes.Time.diff().dt.total_seconds()
    starts=changes[changes.Status.eq('1')&changes.previous.eq('4')&changes.duration.ge(60)]
    clean=start_exposure(laps).set_index(['Driver','LapNumber'],verify_integrity=True)
    rows=[]; reports=[]
    for number,flag in enumerate(starts.itertuples(),1):
        count=0
        for driver,g in laps.groupby('Driver',sort=True):
            following=g[g.LapStartTime.ge(flag.Time)].sort_values('LapNumber')
            if following.empty:
                continue
            first=following.iloc[0]; n=int(first.LapNumber)
            keys=[(driver,n+i) for i in range(4)]
            prior=g[g.LapNumber.eq(n-1)]
            if len(prior)!=1 or not all(k in clean.index for k in keys):
                continue
            p=prior.iloc[0]; current=clean.loc[keys]
            if not pd.notna(p.LapStartTime) or not p.LapStartTime<=flag.Time<=p.Time or '4' not in str(p.TrackStatus):
                continue
            if current.LapStartTime.iloc[0]>flag.Time+pd.Timedelta(seconds=200):
                continue
            if pd.notna(p.PitInTime) or pd.notna(p.PitOutTime):
                continue
            valid=p.Compound in ('SOFT','MEDIUM','HARD') and current.Compound.eq(p.Compound).all() and current.Stint.eq(p.Stint).all()
            ages=np.r_[p.TyreLife,current.TyreLife.to_numpy(float)]
            valid=valid and np.isfinite(ages).all() and np.allclose(np.diff(ages),1)
            valid=valid and current.Position.notna().all() and current.Position.nunique()==1
            clear=flags[flags.Time.gt(flag.Time)&flags.Time.le(current.Time.iloc[-1])].Status.eq('1').all()
            if valid and clear:
                rows.append(dict(restart=number,green_seconds=flag.Time.total_seconds(),sc_seconds=flag.duration,
                    driver=driver,previous_lap=n-1,first_full_lap=n,stint=p.Stint,compound=p.Compound))
                count+=1
        reports.append(dict(restart=number,green_seconds=flag.Time.total_seconds(),sc_seconds=flag.duration,eligible_drivers=count))
    return pd.DataFrame(rows,columns=COLS),reports


def support(source,entries):
    assert entries.season.between(2018,2022).all()
    assert not source.duplicated(['season','round','restart','driver']).any()
    table=source.merge(entries.rename(columns={'code':'driver'}),on=['season','round','driver'],how='left',validate='many_to_one')
    assert table.team.notna().all()
    pairs=[]
    for (season,round_,restart,team),g in table.groupby(['season','round','restart','team'],sort=True):
        roster=entries[entries.season.eq(season)&entries['round'].eq(round_)&entries.team.eq(team)].code
        if len(roster)==2 and set(g.driver)==set(roster):
            pairs.append(dict(season=int(season),round=int(round_),restart=int(restart),team=team))
    paired=pd.DataFrame(pairs,columns=['season','round','restart','team'])
    if paired.empty:
        return dict(paired_team_restarts=0,qualified_restarts=0,qualified_races=0,qualified_team_races=0,gate=False),paired
    counts=paired.groupby(['season','round','restart']).team.transform('count')
    qualified=paired[counts.ge(6)]
    races=len(qualified[['season','round']].drop_duplicates())
    team_races=len(qualified[['season','round','team']].drop_duplicates())
    result=dict(paired_team_restarts=len(paired),qualified_restarts=int(qualified.groupby(['season','round','restart']).ngroups),
        qualified_races=races,qualified_team_races=team_races,gate=bool(races>=12 and team_races>=60))
    return result,paired


def main():
    import fastf1
    import f1.ff1
    entries=load_results().loc[lambda d:d.season.between(2018,2022),['season','round','code','team']]
    fastf1.Cache.offline_mode(True); parts=[]; reports=[]
    for season,round_ in entries[['season','round']].drop_duplicates().itertuples(index=False,name=None):
        report=dict(season=int(season),round=int(round_),available=False)
        try:
            session=cached_session(fastf1,int(season),int(round_))
            laps=pd.DataFrame(session.laps); status=pd.DataFrame(session.track_status)
        except Exception as exc:
            report['error_type']=type(exc).__name__
        else:
            source,restarts=blocks(laps,status)
            parts.append(source.assign(season=season,round=round_))
            report.update(available=True,restarts=restarts,blocks=len(source))
        reports.append(report)
        print('SOURCE',season,round_,report.get('blocks',0),flush=True)
    source=pd.concat(parts,ignore_index=True)
    result,paired=support(source,entries)
    result.update(sessions=reports,blocks=len(source),response_fit=False,winner_fit=False)
    source.to_csv(ROOT/'results/e237_restart_windows.csv',index=False)
    paired.to_csv(ROOT/'results/e237_paired_teams.csv',index=False)
    (ROOT/'results/e237_support_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SUPPORT',json.dumps({k:v for k,v in result.items() if k!='sessions'}),flush=True)


if __name__=='__main__':
    main()
