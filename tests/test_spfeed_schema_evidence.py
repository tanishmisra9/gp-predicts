import json
from experiments.e249_spfeed_schema_evidence import inspect_schema,FIELDS,BLOBS,COMMIT
from f1.data import ROOT


def main():
    fields="\n".join(f"public {n} {n} {{ get; set; }}" for n in FIELDS)
    classes="\n".join(f"public class {n} {{\npublic List<int> pHAM {{ get; set; }}\n}}" for n in FIELDS)
    text="public class Graph {\n"+fields+"\n}\n"+classes
    p='subscriptions.Add("SPFeed"); HubConnection("https://livetiming.formula1.com/signalr")'
    r=inspect_schema(text,p)
    assert r["fields"]==list(FIELDS) and all(v==["pHAM"] for v in r["driver_members"].values())
    assert not r["scores_graph_declared"] and not r["spfeed_scores_declared"]
    changed=inspect_schema(text.replace("public Brake Brake", "public Brake Other"),"")
    assert "Brake" not in changed["fields"] and not changed["subscribes_spfeed"] and not changed["uses_live_hub"]
    artifact=json.loads((ROOT/"results/e249_spfeed_schema_evidence.json").read_text())
    assert artifact["commit"]==COMMIT and {r["path"]:r["blob_sha1"] for r in artifact["receipts"]}==BLOBS
    assert artifact["schema_facts"]["fields"]==list(FIELDS)
    assert artifact["repository_created_at"]=="2019-11-28T08:23:57Z"
    assert artifact["commit_author_date"]==artifact["commit_committer_date"]=="2019-11-28T08:20:25Z"
    assert artifact["post_discovery_audit"] and not any(artifact[k] for k in ("physical_units_verified","score_formula_verified","raw_historical_publication_verified","winner_fit"))
    print("E249 synthetic schema/member/subscription falsification and pinned-metadata/limited-claim checks PASS")


if __name__=="__main__":main()
