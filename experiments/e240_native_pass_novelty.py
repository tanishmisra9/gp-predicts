"""E240 matched-context native pass/repass novelty versus fixed E107 labels."""
import gzip
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from experiments.b2_passing import pass_opportunities
from experiments.e190_traffic_feasibility import PILOT,cached_session
from f1.data import ROOT


def compare(laps,status,markers,span):
    assert not laps.duplicated(['Driver','LapNumber']).any()
    names=laps[['DriverNumber','Driver']].drop_duplicates()
    assert not names.DriverNumber.duplicated().any()
    mapping={str(r.DriverNumber):r.Driver for r in names.itertuples()}
    events=defaultdict(list)
    for m in markers:
        if not m['echo']: continue
        assert m['car'] in mapping
        for other in m['counterparts']:
            assert other in mapping
            events[(mapping[m['car']],mapping[other])].append(float(m['seconds']))
    for values in events.values(): values.sort()
    clean=laps[(laps.LapNumber>1)&laps.Time.notna()&laps.Position.notna()&laps.PitInTime.isna()&laps.PitOutTime.isna()
        &laps.TrackStatus.astype(str).eq('1')&laps.IsAccurate.fillna(False)].set_index(['Driver','LapNumber'],verify_integrity=True)
    flags=status.sort_values('Time',kind='stable')
    assert flags.Time.notna().all()
    base=pass_opportunities(laps)
    assert not base.duplicated(['lap','Driver','ahead']).any()
    rows=[]
    for row in base.itertuples():
        keys=[(d,row.lap+i) for d in (row.Driver,row.ahead) for i in range(3)]
        item=dict(lap=int(row.lap),driver=row.Driver,ahead=row.ahead,gap=float(row.gap),old_pass=int(row.passed),
            activity_eligible=False,native_covered=False,matched=False,forward=[],reverse=[],native_pass=False,confirmed_transient=False,reason='activity')
        if not all(key in clean.index for key in keys):
            rows.append(item);continue
        g=clean.loc[keys]
        times=g.Time.dt.total_seconds().to_numpy().reshape(2,3)
        if not (np.diff(times,axis=1)>0).all():
            rows.append(item);continue
        start,end=times[0,0],times[0,1]; activity_end=float(times[:,-1].max())
        before=flags[flags.Time.le(pd.Timedelta(seconds=start))].tail(1)
        during=flags[flags.Time.gt(pd.Timedelta(seconds=start))&flags.Time.le(pd.Timedelta(seconds=activity_end))]
        if len(before)!=1 or not before.Status.eq('1').all() or not during.Status.eq('1').all():
            item['reason']='caution';rows.append(item);continue
        item.update(activity_eligible=True,start=float(start),end=float(end),activity_end=activity_end,
            native_covered=bool(span[0]<=start and end<=span[1]))
        if not item['native_covered']:
            item['reason']='source_span';rows.append(item);continue
        forward=[t for t in events[(row.Driver,row.ahead)] if start<t<=end]
        reverse=[t for t in events[(row.ahead,row.Driver)] if start<t<=end]
        sustained=False
        for t in forward:
            returns=[r for r in reverse if r>t]
            if returns and returns[0]-t>=5:
                sustained=True
        item.update(matched=True,forward=forward,reverse=reverse,native_pass=bool(forward),
            confirmed_transient=bool(not row.passed and sustained),reason='matched')
        rows.append(item)
    return rows


def summarize(sources):
    rows=[row for source in sources for row in source['rows']]
    eligible=[r for r in rows if r['activity_eligible']]
    matched=[r for r in rows if r['matched']]
    old=sum(r['old_pass'] for r in matched)
    recalled=sum(r['old_pass'] and r['native_pass'] for r in matched)
    novel=sum(r['confirmed_transient'] for r in matched)
    coverage=len(matched)/len(eligible) if eligible else 0.
    recall=recalled/old if old else 0.
    fraction=novel/old if old else 0.
    novel_races=sum(any(r['confirmed_transient'] for r in s['rows']) for s in sources)
    support=bool(len(sources)==3 and all(sum(r['matched'] for r in s['rows'])>=20 for s in sources) and coverage>=.95 and old>=30)
    return dict(old_opportunities=len(rows),activity_opportunities=len(eligible),matched_opportunities=len(matched),coverage=coverage,
        old_positive=old,native_recalled=recalled,recall=recall,confirmed_transients=novel,novel_races=novel_races,novel_fraction=fraction,
        support_gate=support,agreement_gate=bool(support and recall>=.9),
        gate=bool(support and recall>=.9 and novel>=10 and novel_races>=2 and fraction>=.1),winner_fit=False)


def main():
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True)
    saved=json.loads((ROOT/'results/e239_marker_validation.json').read_text()); sources=[]
    for season,round_ in PILOT:
        session=cached_session(fastf1,season,round_)
        records=json.loads(gzip.decompress((ROOT/f'results/e238_2019_{round_}_numeric.json.gz').read_bytes()))
        span=(min(r['seconds'] for r in records),max(r['seconds'] for r in records))
        markers=next(r['markers'] for r in saved['sources'] if r['round']==round_)
        rows=compare(pd.DataFrame(session.laps),pd.DataFrame(session.track_status),markers,span)
        sources.append(dict(season=season,round=round_,span=span,rows=rows))
        print('SOURCE',round_,len(rows),sum(r['matched'] for r in rows),flush=True)
    result=dict(sources=sources,**summarize(sources))
    (ROOT/'results/e240_native_pass_novelty.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',json.dumps({k:v for k,v in result.items() if k!='sources'}),flush=True)


if __name__=='__main__':
    main()
