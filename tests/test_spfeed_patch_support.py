from experiments.e249_spfeed_schema_evidence import FIELDS
from experiments.e250_spfeed_patch_support import analyse


def source(n=10,span=300):
    return [dict(seconds=i*span/(n-1),numeric=[dict(path=["Scores","graph",f,"p"+c,"1"],value=i%2) for f in FIELDS for c in sorted(roster)]) for i in range(n)]


roster={f"D{i:02d}" for i in range(10)}


def main():
    r=analyse(source(),roster)
    assert r["gate"] and all(v==10 for v in r["supported_drivers_by_field"].values())
    assert not analyse(source(9),roster)["gate"]
    assert not analyse(source(span=299.999),roster)["gate"]
    constant=source()
    for rec in constant:
        for item in rec["numeric"]:item["value"]=1
    assert not analyse(constant,roster)["gate"]
    rows=source();rows[0]["numeric"].append(dict(path=["Scores","graph","Brake","pD00","[]"],value=10))
    assert analyse(rows,roster)["gate"] and analyse(rows,roster)["rejection_counts"]=={"ambiguous_array_index":1}
    rows=source();rows[0]["numeric"][0]["path"][3]="pXXX"
    assert not analyse(rows,roster)["gate"]
    rows=source();rows[0]["numeric"][0]["value"]=1.5
    assert not analyse(rows,roster)["gate"]
    assert not analyse([],roster)["gate"]
    print("E250 count/span boundaries/variable/constant/ambiguousinit/unknownidentity/nonnumericinteger/empty checks PASS")


if __name__=="__main__":main()
