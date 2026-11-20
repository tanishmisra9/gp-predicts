"""E249 pinned primary client-schema evidence, never a forecast input."""
import base64
import hashlib
import json
import re
import subprocess

from f1.data import ROOT

REPO="claudiopizzillo/F1client"
COMMIT="ee9730b34c013fd7f0b3c335e82dbb0de650f54e"
BLOBS={"TimingData.cs":"6f509b715b81579f904f52a0307c36633bcf00c3","Program.cs":"3200a6cc712911b524449c99df9ba5d8ce246897"}
FIELDS=("Steering","GforceLat","GforceLong","Brake","Performance","Throttle")


def inspect_schema(schema,program):
    graph=re.search(r"public class Graph\s*\{(.*?)\n\s*\}",schema,re.S)
    assert graph is not None
    present=[name for name in FIELDS if re.search(r"public "+name+r" "+name+r" \{ get; set; \}",graph.group(1))]
    drivers={}
    for name in present:
        block=re.search(r"public class "+name+r"\s*\{(.*?)\n\s*\}",schema,re.S)
        assert block is not None
        drivers[name]=sorted(re.findall(r"public List<int\??> (p[A-Z]{3}) \{ get; set; \}",block.group(1)))
    return dict(fields=present,driver_members=drivers,
                subscribes_spfeed='subscriptions.Add("SPFeed")' in program,
                uses_live_hub='HubConnection("https://livetiming.formula1.com/signalr")' in program,
                scores_graph_declared=bool(re.search(r"public class Scores\s*\{\s*public Graph graph",schema)),
                spfeed_scores_declared=bool(re.search(r"public class SPFeed\s*\{\s*public Scores Scores",schema)))


def get(path):
    return json.loads(subprocess.run(["gh","api",path],check=True,capture_output=True,text=True).stdout)


def main():
    repo=get(f"repos/{REPO}")
    commit=get(f"repos/{REPO}/commits/{COMMIT}")
    tree=get(f"repos/{REPO}/git/trees/{commit['commit']['tree']['sha']}")
    sources={};receipts=[]
    for path,sha in BLOBS.items():
        assert any(t["path"]==path and t["sha"]==sha and t["type"]=="blob" for t in tree["tree"])
        blob=get(f"repos/{REPO}/git/blobs/{sha}")
        assert blob["encoding"]=="base64"
        raw=base64.b64decode(blob["content"])
        assert hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()==sha
        sources[path]=raw.decode("utf-8-sig")
        receipts.append(dict(path=path,blob_sha1=sha,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),
                             url=f"https://github.com/{REPO}/blob/{COMMIT}/{path}"))
    facts=inspect_schema(sources["TimingData.cs"],sources["Program.cs"])
    assert facts["fields"]==list(FIELDS) and all(len(v)>=10 for v in facts["driver_members"].values())
    assert all(facts[k] for k in ("subscribes_spfeed","uses_live_hub","scores_graph_declared","spfeed_scores_declared"))
    report=dict(repository=REPO,repository_created_at=repo["created_at"],commit=COMMIT,
                commit_author_date=commit["commit"]["author"]["date"],commit_committer_date=commit["commit"]["committer"]["date"],
                commit_signature_verified=commit["commit"]["verification"]["verified"],
                receipts=receipts,schema_facts=facts,physical_units_verified=False,score_formula_verified=False,
                raw_historical_publication_verified=False,winner_fit=False,post_discovery_audit=True)
    (ROOT/"results/e249_spfeed_schema_evidence.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("receipts","schema_facts")},indent=2))
    print("fields",facts["fields"],"driver counts",{k:len(v) for k,v in facts["driver_members"].items()})


if __name__=="__main__":main()
