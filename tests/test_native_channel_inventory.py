import json
from experiments.e247_native_channel_inventory import inventory


def record(value,extra=True):
    cars={str(n):{"Channels":{"0":10000,"2":200,"3":6,"4":100,"5":0,"45":12,**({"99":value} if extra else {})}} for n in range(10)}
    return "00:00:00.000"+json.dumps({"Entries":[{"Cars":cars}]})


def main():
    roster={str(n) for n in range(10)}
    rows=[record(n%2) for n in range(100)]
    r=inventory(rows,roster,json.loads)
    assert r["gate"] and r["extra_channels"]==["99"] and r["supported_extra_cars"]=={"99":10}
    assert not inventory(rows[:99],roster,json.loads)["gate"]
    assert not inventory([record(1) for _ in range(100)],roster,json.loads)["gate"]
    assert not inventory([record(0,False) for _ in range(100)],roster,json.loads)["gate"]
    r=inventory([record("uninterpreted") for _ in range(100)],roster,json.loads)
    assert not r["gate"] and all(x["numeric"]==0 for x in r["rows"] if x["channel"]=="99")
    r=inventory([record(True) for _ in range(100)],roster,json.loads)
    assert not r["gate"]
    assert inventory([],roster,json.loads)["rows"]==[]
    try:inventory(rows,{str(n) for n in range(1,11)},json.loads)
    except AssertionError:pass
    else:raise AssertionError("unknown car accepted")
    print("E247 known/extra/constant/nonnumeric/boolean/countboundary/empty/identity checks PASS")


if __name__=="__main__":main()
