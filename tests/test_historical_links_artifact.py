"""E254 scalar CSV projection and independent disjoint-set topology audit."""
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict,Counter
from datetime import datetime

from f1.data import ROOT,DATA


def extract(path,years):
    years=set(years);rows=[];groups=defaultdict(list);seen=set();counts=Counter();valid=Counter()
    with path.open() as stream:
        for raw in csv.DictReader(stream):
            year=int(raw["season"])
            if year not in years:continue
            round_=int(raw["round"]);driver=raw["driver"];team=raw["team"]
            assert driver and team and (year,round_,driver) not in seen
            seen.add((year,round_,driver))
            assert datetime.fromisoformat(raw["date"])<datetime(2020,1,1)
            q1=float(raw["q1"] or "nan")
            good=math.isfinite(q1) and q1>0
            counts[year]+=1;valid[year]+=int(good)
            groups[(year,round_,team)].append((driver,good))
    for (year,round_,team),group in sorted(groups.items()):
        if len(group)==2 and all(good for driver,good in group):
            a,b=sorted(driver for driver,good in group)
            rows.append(dict(season=year,round=round_,team=team,a=a,b=b))
    coverage={str(y):dict(rows=counts[y],valid=valid[y],fraction=valid[y]/counts[y]) for y in sorted(counts)}
    appearances=defaultdict(set)
    for row in rows:appearances[(row["a"],row["b"])].add((row["season"],row["round"]))
    edges={p for p,events in appearances.items() if len(events)>=3}
    return rows,coverage,edges,seen


def components(edges,reference):
    nodes=set(reference)|set(d for p in edges for d in p)
    parent={d:d for d in nodes}
    def find(d):
        while parent[d]!=d:d=parent[d]
        return d
    for a,b in sorted(edges):parent[find(a)]=find(b)
    groups=defaultdict(list)
    for d in sorted(nodes):groups[find(d)].append(d)
    paired=[list(p) for p in itertools.combinations(sorted(reference),2) if find(p[0])==find(p[1])]
    return dict(nodes=len(nodes),edges=len(edges),components=sorted(groups.values()),reference_connected_pairs=paired,
                reference_pair_count=len(reference)*(len(reference)-1)//2,
                reference_pair_fraction=len(paired)/(len(reference)*(len(reference)-1)//2))


def main():
    report=json.loads((ROOT/"results/e254_historical_links.json").read_text())
    old=DATA/"results_2014_2017.csv";modern=DATA/"results.csv"
    for p in (old,modern):assert hashlib.sha256(p.read_bytes()).hexdigest()==report["hashes"][p.name]
    a,ca,ea,_=extract(old,range(2014,2018));b,cb,eb,seen=extract(modern,range(2018,2020))
    last=max(round_ for year,round_,driver in seen if year==2019)
    reference=sorted(driver for year,round_,driver in seen if (year,round_)==(2019,last))
    assert len(reference)==20 and reference==report["reference"]
    assert a==report["old_pair_observations"] and b==report["modern_pair_observations"]
    assert ca==report["old_coverage"] and cb==report["modern_coverage"]
    assert [list(p) for p in sorted(ea)]==report["older_edges"]
    assert [list(p) for p in sorted(eb)]==report["modern_edges"]
    base=components(eb,reference);extended=components(ea|eb,reference)
    assert base==report["base"] and extended==report["extended"]
    gain=extended["reference_pair_fraction"]-base["reference_pair_fraction"]
    source=all(row["fraction"]>=.9 for row in list(ca.values())+list(cb.values())) and len(ea)>=10
    assert gain==report["connected_pair_fraction_gain"] and source==report["source_gate"]
    assert report["gate"]==(source and gain>=.25)
    assert report["time_constant_driver_assumption"] and not report["current_skill_identified"] and not report["winner_fit"]
    print("E254 independent scalar qualifying/identity/year/3-event edges/union-find/all190pairs/hash/gate audit PASS")


if __name__=="__main__":main()
