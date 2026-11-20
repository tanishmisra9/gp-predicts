"""E256 pinned primary pit-reader mapping plus saved 2019 feed metadata."""
import ast
import base64
import hashlib
import json
import subprocess

from f1.data import ROOT

REPO="br-g/openf1"
COMMIT="853ae3398a243487c52306b99093f067c951cb50"
BLOB="a57909bb0c6869ce3f31be93f289f275d42a7079"
PATH="src/openf1/services/ingestor_livetiming/core/processing/collections/pit.py"


def inspect_reader(source):
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="PitCollection")
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="process_message")
    branch=next(n for n in method.body if isinstance(n,ast.If))
    result={}
    while isinstance(branch,ast.If):
        assert isinstance(branch.test,ast.Compare) and ast.unparse(branch.test.left)=="message.topic"
        assert len(branch.test.comparators)==1 and isinstance(branch.test.comparators[0],ast.Constant)
        topic=branch.test.comparators[0].value
        block=ast.Module(body=branch.body,type_ignores=[])
        constructors=[n for n in ast.walk(block) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="Pit"]
        assert len(constructors)==1
        stop=next(k.value for k in constructors[0].keywords if k.arg=="stop_duration")
        inputs={name:set() for name in ("lane_duration","stop_duration")}
        for node in ast.walk(block):
            if not isinstance(node,ast.Assign) or len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name):continue
            name=node.targets[0].id
            if name not in inputs or not isinstance(node.value,ast.Call) or ast.unparse(node.value.func)!="float":continue
            for leaf in ast.walk(node.value):
                if isinstance(leaf,ast.Subscript) and isinstance(leaf.slice,ast.Constant) and isinstance(leaf.slice.value,str):inputs[name].add(leaf.slice.value)
        result[topic]=dict(stop_argument=ast.unparse(stop),lane_input_keys=sorted(inputs["lane_duration"]),stop_input_keys=sorted(inputs["stop_duration"]))
        branch=branch.orelse[0] if len(branch.orelse)==1 else None
    return result


def main():
    endpoint=f"repos/{REPO}/git/blobs/{BLOB}"
    payload=json.loads(subprocess.run(["gh","api",endpoint],check=True,capture_output=True,text=True).stdout)
    assert payload["encoding"]=="base64"
    raw=base64.b64decode(payload["content"])
    assert hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()==BLOB
    reader=inspect_reader(raw.decode())
    assert set(reader)=={"PitStop","PitStopSeries","PitLaneTimeCollection"}
    for name in ("PitStop","PitStopSeries"):
        assert reader[name]["stop_argument"]=="stop_duration" and "PitStopTime" in reader[name]["stop_input_keys"]
    assert reader["PitLaneTimeCollection"]["stop_argument"]=="None" and reader["PitLaneTimeCollection"]["lane_input_keys"]==["Duration"]
    path=ROOT/"results/e229_pit_feed_index.json"
    old=json.loads(path.read_text());sources=[]
    service={name for name,r in reader.items() if r["stop_argument"]!="None"}
    for source in old["sessions"]:
        names=set(source["feeds"])
        sources.append(dict(season=source["season"],round=source["round"],lane_present="PitLaneTimeCollection" in names,
                            service_topics_present=sorted(names&service)))
    result=dict(reader=reader,service_topics=sorted(service),sources=sources,
                commit=COMMIT,blob_sha1=BLOB,source_sha256=hashlib.sha256(raw).hexdigest(),index_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                primary_url=f"https://github.com/{REPO}/blob/{COMMIT}/{PATH}",
                raw_historical_service_fields_checked=False,historical_units_verified=False,winner_fit=False,post_inspection_audit=True)
    (ROOT/"results/e256_pit_reader_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
