"""E200 following exposure-source audit, no race-pace response or forecast."""
import json
from unittest.mock import patch

import pandas as pd

from experiments.e190_traffic_feasibility import cached_session,disjoint_pairs,start_exposure
from experiments.e199_leader_pressure import prepare_stream
from f1.data import ROOT,load_results

META=['Driver','LapNumber','LapStartTime','Stint','Compound','TyreLife','Position','proxy_close','end_proxy_close','stream_time','live_position','interval','interval_observed','interval_age','fresh','stream_close']


def following(source,targets):
    right=source[['Driver','Time','Position','gap','observed']].rename(columns={'Time':'stream_time','Position':'live_position','gap':'interval','observed':'interval_observed'})
    out=pd.merge_asof(targets.sort_values('LapStartTime'),right.sort_values('stream_time'),by='Driver',left_on='LapStartTime',right_on='stream_time',direction='backward')
    out['interval_age']=(out.LapStartTime-out.interval_observed).dt.total_seconds()
    out['fresh']=out.interval.gt(0)&out.interval.notna()&out.live_position.gt(1)&out.interval_age.between(0,30)
    out['stream_close']=out.interval.le(1.5).where(out.fresh).astype('boolean')
    return out


def count_support(pairs,entries):
    rows=[]
    for (round_,team),_ in entries.groupby(['round','team'],sort=True):
        g=pairs[pairs['round'].eq(round_)&pairs.team.eq(team)].sort_values(['completion2','driver','lap1'])
        entering=int(g.entering.sum()); leaving=int((~g.entering).sum())
        full=len(g)>=8 and entering>=2 and leaving>=2
        halves=[]
        for offset in (0,1):
            half=g.iloc[offset::2]
            halves.append(len(half)>=4 and half.entering.any() and (~half.entering).any())
        rows.append(dict(season=2019,round=int(round_),team=team,pairs=len(g),entering=entering,leaving=leaving,full=bool(full),half0=bool(halves[0]),half1=bool(halves[1])))
    table=pd.DataFrame(rows)
    full_counts=table.groupby('round').full.transform('sum'); table['valid_full']=table.full&full_counts.ge(6)
    table['valid_halves']=True
    for name in ('half0','half1'):
        table['valid_halves']&=table[name]&table.groupby('round')[name].transform('sum').ge(6)
    table['past_full']=table.sort_values('round').groupby('team').valid_full.cumsum()-table.valid_full.astype(int)
    temporal=table.valid_full&table.past_full.ge(3)
    return table,int(table.groupby('round').valid_full.any().sum()),int(table.valid_halves.sum()),int(temporal.sum())


def main():
    import fastf1
    import f1.ff1
    from fastf1 import _api
    from fastf1.internals.f1account import LivetimingAuthAdapter
    fastf1.Cache.offline_mode(True)
    entries=load_results().loc[lambda d:d.season.eq(2019),['season','round','code','team']]
    assert entries['round'].nunique()==21
    reports=[]; all_rows=[]; all_pairs=[]
    for round_,event in entries.groupby('round',sort=True):
        report=dict(season=2019,round=int(round_),available=False)
        try:
            session=cached_session(fastf1,2019,int(round_)); raw=pd.DataFrame(session.laps)
            with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('HTTP forbidden in interval pilot')), patch.object(LivetimingAuthAdapter,'_get_token',return_value=None):
                _,stream,_=_api._extended_timing_data(session.api_path)
            source=prepare_stream(stream,raw[['DriverNumber','Driver']].drop_duplicates(),field='IntervalToPositionAhead')
            clean=start_exposure(raw); targets=clean[clean.Position.gt(1)].copy(); targets['proxy_close']=targets.close
            line=raw[raw.Time.notna()&raw.Position.notna()].sort_values(['LapNumber','Position']).copy()
            end_gap=(line.Time-line.groupby('LapNumber').Time.shift()).dt.total_seconds()
            line['end_proxy_close']=end_gap.between(0,1.5)
            targets=targets.merge(line[['Driver','LapNumber','end_proxy_close']],on=['Driver','LapNumber'],validate='one_to_one')
            current=following(source,targets)
            valid=current[current.fresh].copy(); valid['close']=valid.stream_close.astype(bool)
            pairs=disjoint_pairs(valid)
            pairs=pairs.merge(valid[['Driver','LapNumber','Time']].rename(columns={'Driver':'driver','LapNumber':'lap2','Time':'completion2'}),on=['driver','lap2'],validate='many_to_one')
            pairs=pairs.merge(event[['code','team']].rename(columns={'code':'driver'}),on='driver',how='left',validate='many_to_one')
            assert pairs.team.notna().all()
            report.update(available=True,targets=len(current),matched=len(valid),coverage=float(len(valid)/len(current)) if len(current) else 0.,pairs=len(pairs))
            all_rows.append(current[META].assign(season=2019,round=int(round_)))
            all_pairs.append(pairs.assign(season=2019,round=int(round_)))
        except Exception as exc:
            report['error_type']=type(exc).__name__
        reports.append(report); print(json.dumps(report),flush=True)
    table=pd.concat(all_rows,ignore_index=True) if all_rows else pd.DataFrame(columns=META+['season','round'])
    pairs=pd.concat(all_pairs,ignore_index=True) if all_pairs else pd.DataFrame(columns=['driver','lap1','lap2','entering','stint','completion2','team','season','round'])
    matched=table[table.fresh]; covered=sum(r.get('coverage',0)>=.8 for r in reports)
    confusion={}
    for name in ('proxy_close','end_proxy_close'):
        confusion[name]={f'{int(old)}{int(new)}':int((matched[name].eq(old)&matched.stream_close.eq(new)).sum()) for old in (False,True) for new in (False,True)}
    primary=confusion['proxy_close']; disagreement=(primary['01']+primary['10'])/len(matched) if len(matched) else 0.
    support,races,halves,temporal=count_support(pairs,entries)
    gate=covered>=15 and len(matched)>=10000 and disagreement>=.05 and primary['01']>=100 and primary['10']>=100 and races>=12 and halves>=60 and temporal>=60
    result=dict(sessions=reports,targets=len(table),matched=len(matched),covered_races=covered,confusion=confusion,disagreement=disagreement,pairs=len(pairs),valid_response_races=races,half_opportunities=halves,temporal_opportunities=temporal,gate=bool(gate))
    (ROOT/'results/e200_following_interval_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    table.to_csv(ROOT/'results/e200_following_interval_exposures.csv',index=False)
    pairs.to_csv(ROOT/'results/e200_following_interval_pairs.csv',index=False)
    support.to_csv(ROOT/'results/e200_following_interval_support.csv',index=False)
    print('SOURCE GATE',bool(gate),flush=True)


if __name__=='__main__':
    main()
