"""E254 existing pre-2020 Q1 teammate-link topology, no pace/skill fit."""
import hashlib
import itertools
import json

import numpy as np
import pandas as pd

from f1.data import ROOT,DATA,load_results


def links(frame,start,end):
    d=frame[frame.season.between(start,end)].copy()
    assert not d.duplicated(["season","round","driver"]).any()
    assert d.driver.notna().all() and d.team.notna().all()
    assert pd.to_datetime(d.date).lt(pd.Timestamp("2020-01-01")).all()
    valid=np.isfinite(d.q1)&d.q1.gt(0)
    coverage={str(y):dict(rows=len(g),valid=int(valid.loc[g.index].sum()),fraction=float(valid.loc[g.index].mean())) for y,g in d.groupby("season",sort=True)}
    rows=[]
    for (season,round_,team),g in d.groupby(["season","round","team"],sort=True):
        if len(g)!=2 or not valid.loc[g.index].all():continue
        a,b=sorted(g.driver)
        rows.append(dict(season=int(season),round=int(round_),team=team,a=a,b=b))
    return rows,coverage


def qualified(rows):
    counts={}
    for r in rows:
        pair=(r["a"],r["b"]);counts[pair]=counts.get(pair,0)+1
    return {pair for pair,n in counts.items() if n>=3}


def topology(edges,reference):
    nodes=set(reference)|{d for pair in edges for d in pair}
    graph={n:set() for n in nodes}
    for a,b in edges:graph[a].add(b);graph[b].add(a)
    components=[];remaining=set(nodes)
    while remaining:
        start=min(remaining);seen={start};queue=[start]
        while queue:
            for n in sorted(graph[queue.pop()]):
                if n not in seen:seen.add(n);queue.append(n)
        remaining-=seen;components.append(sorted(seen))
    membership={d:i for i,c in enumerate(components) for d in c}
    pairs=list(itertools.combinations(sorted(reference),2))
    connected=[list(p) for p in pairs if membership[p[0]]==membership[p[1]]]
    return dict(nodes=len(nodes),edges=len(edges),components=components,reference_connected_pairs=connected,
                reference_pair_count=len(pairs),reference_pair_fraction=len(connected)/len(pairs) if pairs else 0.)


def main():
    oldpath=DATA/"results_2014_2017.csv";currentpath=DATA/"results.csv"
    old=pd.read_csv(oldpath,parse_dates=["date"]).reset_index(drop=True)
    recent=load_results().loc[lambda d:d.season.between(2018,2019)].reset_index(drop=True)
    assert set(old.season)=={2014,2015,2016,2017} and set(recent.season)=={2018,2019}
    reference=sorted(recent[recent.season.eq(2019)&recent["round"].eq(recent[recent.season.eq(2019)]["round"].max())].driver)
    assert len(reference)==20 and len(set(reference))==20
    old_rows,old_coverage=links(old,2014,2017)
    modern_rows,modern_coverage=links(recent,2018,2019)
    older=qualified(old_rows);modern=qualified(modern_rows)
    base=topology(modern,reference);extended=topology(modern|older,reference)
    gain=extended["reference_pair_fraction"]-base["reference_pair_fraction"]
    source_gate=all(r["fraction"]>=.9 for r in list(old_coverage.values())+list(modern_coverage.values())) and len(older)>=10
    report=dict(reference=reference,old_coverage=old_coverage,modern_coverage=modern_coverage,
                old_pair_observations=old_rows,modern_pair_observations=modern_rows,
                older_edges=[list(p) for p in sorted(older)],modern_edges=[list(p) for p in sorted(modern)],
                base=base,extended=extended,connected_pair_fraction_gain=gain,source_gate=bool(source_gate),
                gate=bool(source_gate and gain>=.25),time_constant_driver_assumption=True,
                current_skill_identified=False,winner_fit=False,
                hashes={oldpath.name:hashlib.sha256(oldpath.read_bytes()).hexdigest(),currentpath.name:hashlib.sha256(currentpath.read_bytes()).hexdigest()})
    (ROOT/"results/e254_historical_links.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("old_pair_observations","modern_pair_observations","base","extended")},indent=2))
    print("TOPOLOGY",{k:dict(components=len(v["components"]),connected=len(v["reference_connected_pairs"]),total=v["reference_pair_count"],fraction=v["reference_pair_fraction"]) for k,v in (("base",base),("extended",extended))})


if __name__=="__main__":main()
