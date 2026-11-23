"""E267 explicit-state provenance, no unknown-to-green inference."""
import json

from experiments.e267_initial_track_status import coverage, project

S = 1_000_000_000


def main():
    def record(t, **row):
        return f"00:00:{t:02}.000" + json.dumps(row)
    raw = (record(5, Status="1", Message="unused") + "\r\n" + record(6, Message="not read") +
           "\r\n" + record(10, Status="2") + "\r\n" + record(12, Status="1")).encode()
    records = project(raw)
    assert len(records) == 4 and records[-1]["time_ns"] == 12*S
    assert not records[1]["has_status"] and all("Message" not in r for r in records)
    assert not coverage(records, 0, 4*S)["complete"]
    assert not coverage(records, 0, 7*S)["native_green"]
    assert coverage(records, 5*S, 9*S)["native_green"]
    assert coverage(records, 6*S, 9*S)["native_green"]
    assert coverage(records, 5*S, 10*S) == dict(complete=True, native_green=False, codes="12")
    assert coverage(records, 12*S, 13*S)["native_green"]
    assert not coverage(records, None, 9*S)["complete"]
    assert not coverage([], 1, 2)["native_green"]
    assert not coverage(records, 2, 2)["complete"]
    unknown = project((record(0, Status="1") + "\n" + record(2, Status="") + "\n" + record(4, Status="1")).encode())
    assert not coverage(unknown, 0, 5*S)["complete"]
    assert not coverage(unknown, 3*S, 5*S)["complete"]
    assert coverage(unknown, 4*S, 5*S)["native_green"]
    duplicates = project((record(0, Status="2") + "\n" + record(0, Status="1")).encode())
    assert coverage(duplicates, 0, S)["native_green"]
    assert project(b"") == []
    assert project(b"\xef\xbb\xbf" + record(0, Status="1").encode())[0]["time_ns"] == 0
    for body in [b"bad", (record(2, Status="1")+"\n"+record(1, Status="1")).encode(),
                 b"00:00:00.000[]", b"00:00:00.000{\"Status\":true}", b"00:00:00.000{"]:
        try: project(body)
        except (AssertionError, json.JSONDecodeError): pass
        else: raise AssertionError("malformed source accepted")
    changed = raw.replace(b"not read", b"different ignored text")
    assert project(changed) == records
    print("E267 explicit/unknown/partial/clock/duplicate/boundary/no-final-CRLF/no-text tests PASS")


if __name__ == "__main__":
    main()
