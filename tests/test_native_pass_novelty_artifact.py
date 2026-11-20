"""E240 independent raw lap-line enumeration and matched pass/repass audit."""
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

import f1.ff1
from experiments.e190_traffic_feasibility import cached_session


def good(row,time=True):
    return row.LapNumber>1 and pd.notna(row.Position) and (not time or pd.notna(row.Time)) and pd.isna(row.PitInTime) and pd.isna(row.PitOutTime) and str(row.TrackStatus)=='1' and pd.notna(row.IsAccurate) and bool(row.IsAccurate)


def test_artifacts():
    result=json.loads(Path('results/e240_native_pass_novelty.json').read_text())
    markers=json.loads(Path('results/e239_marker_validation.json').read_text())
    assert [(s['season'],s['round']) for s in result['sources']]==[(2019,5),(2019,12),(2019,14)]
    fastf1.Cache.offline_mode(True); all_rows=[]
    for report in result['sources']:
        session=cached_session(fastf1,2019,report['round'])
        laps=pd.DataFrame(session.laps); status=pd.DataFrame(session.track_status).sort_values('Time',kind='stable')
        lookup={(r.Driver,int(r.LapNumber)):r for r in laps.itertuples()}
        assert len(lookup)==len(laps)
        numbers={str(r.DriverNumber):r.Driver for r in laps.itertuples()}
        groups=defaultdict(list)
        for r in laps.itertuples():
            if good(r): groups[int(r.LapNumber)].append(r)
        old=[]
        for lap,rows in sorted(groups.items()):
            ordered=sorted(rows,key=lambda r:int(r.Position))
            for ahead,follower in zip(ordered,ordered[1:]):
                gap=(follower.Time-ahead.Time).total_seconds()
                if follower.Position-ahead.Position!=1 or not 0<=gap<=1.5: continue
                a=lookup.get((follower.Driver,lap+1)); b=lookup.get((ahead.Driver,lap+1))
                if a is None or b is None or not good(a,False) or not good(b,False): continue
                old.append(dict(lap=lap,driver=follower.Driver,ahead=ahead.Driver,gap=float(gap),old_pass=int(a.Position<b.Position)))
        path=Path(f"results/e238_2019_{report['round']}_numeric.json.gz")
        marker=next(s for s in markers['sources'] if s['round']==report['round'])
        assert marker['projection_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        records=json.loads(gzip.decompress(path.read_bytes()))
        span=[min(r['seconds'] for r in records),max(r['seconds'] for r in records)]
        assert report['span']==span
        pair_events=defaultdict(list)
        for m in marker['markers']:
            if m['echo']:
                for counterpart in m['counterparts']:
                    pair_events[(numbers[m['car']],numbers[counterpart])].append(m['seconds'])
        expected=[]
        for item in old:
            row=dict(item,activity_eligible=False,native_covered=False,matched=False,forward=[],reverse=[],native_pass=False,confirmed_transient=False,reason='activity')
            sequence=[[lookup.get((d,item['lap']+i)) for i in range(3)] for d in (item['driver'],item['ahead'])]
            active=all(r is not None and good(r) for triple in sequence for r in triple)
            if not active:
                expected.append(row); continue
            clock=[[r.Time.total_seconds() for r in triple] for triple in sequence]
            if not all(a<b for triple in clock for a,b in zip(triple,triple[1:])):
                expected.append(row); continue
            start,end=clock[0][:2]; last=max(triple[-1] for triple in clock)
            prior=[r for r in status.itertuples() if r.Time.total_seconds()<=start]
            cautions=[r for r in status.itertuples() if start<r.Time.total_seconds()<=last and r.Status!='1']
            if not prior or prior[-1].Status!='1' or cautions:
                row['reason']='caution';expected.append(row);continue
            covered=span[0]<=start and end<=span[1]
            row.update(activity_eligible=True,start=start,end=end,activity_end=last,native_covered=covered)
            if not covered:
                row['reason']='source_span';expected.append(row);continue
            forward=sorted(t for t in pair_events[(item['driver'],item['ahead'])] if start<t<=end)
            reverse=sorted(t for t in pair_events[(item['ahead'],item['driver'])] if start<t<=end)
            confirmed=False
            if not item['old_pass']:
                for t in forward:
                    later=sorted(r for r in reverse if r>t)
                    if later and later[0]-t>=5: confirmed=True
            row.update(matched=True,forward=forward,reverse=reverse,native_pass=bool(forward),confirmed_transient=confirmed,reason='matched')
            expected.append(row)
        key=lambda r:(r['lap'],r['driver'],r['ahead'])
        actual_map={key(r):r for r in report['rows']}; expected_map={key(r):r for r in expected}
        assert actual_map.keys()==expected_map.keys()
        for k,actual in actual_map.items():
            wanted=expected_map[k]
            assert actual.keys()==wanted.keys()
            for field,value in wanted.items():
                if isinstance(value,float):
                    np.testing.assert_allclose(actual[field],value,atol=1e-10,rtol=0)
                else:
                    assert actual[field]==value,(k,field,actual[field],value)
        all_rows.extend(expected)
        print(report['round'],'old',len(expected),'matched',sum(r['matched'] for r in expected),'transient',sum(r['confirmed_transient'] for r in expected))
    eligible=[r for r in all_rows if r['activity_eligible']]; matched=[r for r in all_rows if r['matched']]
    old=sum(r['old_pass'] for r in matched); recalled=sum(r['old_pass'] and r['native_pass'] for r in matched)
    novel=sum(r['confirmed_transient'] for r in matched)
    coverage=len(matched)/len(eligible) if eligible else 0.
    recall=recalled/old if old else 0.; fraction=novel/old if old else 0.
    races=sum(any(r['confirmed_transient'] for r in s['rows']) for s in result['sources'])
    support=bool(len(result['sources'])==3 and all(sum(r['matched'] for r in s['rows'])>=20 for s in result['sources']) and coverage>=.95 and old>=30)
    metrics=dict(old_opportunities=len(all_rows),activity_opportunities=len(eligible),matched_opportunities=len(matched),coverage=coverage,
        old_positive=old,native_recalled=recalled,recall=recall,confirmed_transients=novel,novel_races=races,novel_fraction=fraction,
        support_gate=support,agreement_gate=bool(support and recall>=.9),gate=bool(support and recall>=.9 and novel>=10 and races>=2 and fraction>=.1),winner_fit=False)
    assert all(result[k]==v for k,v in metrics.items())


if __name__=='__main__':
    test_artifacts()
    print('E240 independent raw lap-line labels/activity/status/windows/first-return/count/gate audit PASS')
