"""E261 fixed technical-source audit, no third-party code execution or race fit."""
import ast
import base64
import csv
import hashlib
import io
import json
import math
import subprocess
from f1.data import ROOT

REPO="aminembarek2/f1-predict"
PIN="bd234c97826b60572e0b953663165ee5bc3a4645"
BLOBS={"f1pred/evaluation/backtest.py":"8d60690b072cc450f39b5529bd6e4d9c1a7d1853",
       "f1pred/sim/race.py":"42ddcba01716741c3f6e4626035ec4f8c7aed700",
       "results/backtest_tuning_strict/leaderboard.csv":"e09e703d0cb99ac61f198dc4c260eb52ac450f20"}


def skip_policy(source):
    tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="walk_forward")
    return any(isinstance(n,ast.If) and isinstance(n.test,ast.UnaryOp) and isinstance(n.test.op,ast.Not)
               and isinstance(n.test.operand,ast.Attribute) and n.test.operand.attr=="scoreable"
               and isinstance(n.test.operand.value,ast.Name) and n.test.operand.value.id=="alignment"
               and any(isinstance(s,ast.Continue) for s in n.body)
               and any(isinstance(s,ast.Constant) and s.value=="winner_not_named" for s in ast.walk(n))
               for n in ast.walk(fn))


def toy():
    # Three named-winner races (two correct); two unseen-winner races are misses.
    eligible=[True,True,True,False,False];hits=[1,0,1,0,0]
    full=sum(hits)/len(hits)
    selected=sum(h for h,ok in zip(hits,eligible) if ok)/sum(eligible)
    assert full==.4 and selected==2/3 and selected>full
    return dict(all_races=5,known_winner_races=3,hits=2,all_race_accuracy=full,conditional_accuracy=selected)


def main():
    tree=json.loads(subprocess.check_output(["gh","api",f"repos/{REPO}/git/trees/{PIN}?recursive=1"]))
    assert not tree.get("truncated",False)
    paths={r["path"]:r["sha"] for r in tree["tree"] if r["type"]=="blob"}
    assert all(paths[path]==sha for path,sha in BLOBS.items())
    saved={};decoded={}
    for path,sha in BLOBS.items():
        response=json.loads(subprocess.check_output(["gh","api",f"repos/{REPO}/git/blobs/{sha}"]))
        body=base64.b64decode(response["content"])
        actual=hashlib.sha1(b"blob "+str(len(body)).encode()+b"\0"+body).hexdigest()
        assert actual==response["sha"]==sha
        decoded[path]=body.decode()
        saved[path]=dict(git_blob=sha,sha256=hashlib.sha256(body).hexdigest(),bytes=len(body))
    policy=skip_policy(decoded["f1pred/evaluation/backtest.py"])
    assert policy
    # A mutated policy without the skip must be detected; do not execute source.
    changed=decoded["f1pred/evaluation/backtest.py"].replace("if not alignment.scoreable:","if alignment.scoreable:")
    assert not skip_policy(changed)
    rows=list(csv.DictReader(io.StringIO(decoded["results/backtest_tuning_strict/leaderboard.csv"])))
    selected={r["model"]:r for r in rows if r["model"] in ("strength_mc","championship")}
    counts={}
    for name,r in selected.items():
        n=int(float(r["n_races"]));p=float(r["top1"]);hits=round(n*p)
        assert n==87 and hits==43 and abs(p-hits/n)<1e-14
        z=1.959963984540054;den=1+z*z/n
        center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        # Primary implementation rounds its normal quantile to 1.959963985.
        assert abs(center-half-float(r["top1_ci_low"]))<1e-9
        assert abs(center+half-float(r["top1_ci_high"]))<1e-9
        counts[name]=dict(races=n,hits=hits,accuracy=p,wilson95=[center-half,center+half])
    assert len(counts)==2
    result=dict(repo=REPO,pin=PIN,blobs=saved,reported_metrics=counts,winner_not_named_skip_present=policy,
                denominator_counterexample=toy(),external_code_executed=False,forecast_replicated=False,
                proves_universal_accuracy_ceiling=False,winner_fit=False)
    report=ROOT/"results/e261_creator_audit.json"
    report.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
