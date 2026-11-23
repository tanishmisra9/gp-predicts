"""E264 scope/coverage/stop audit; no denied or unknown endpoint requests."""
import hashlib
import json
from collections import Counter
from unittest.mock import patch
from f1.data import ROOT
from experiments.e190_traffic_feasibility import cached_session


def main():
    import fastf1
    import f1.ff1
    fastf1.Cache.offline_mode(True)
    reports=json.loads((ROOT/"results/e264_position_status.json").read_text())
    inventory=json.loads((ROOT/"results/e212_source_status.json").read_text())
    result=json.loads((ROOT/"results/e264_position_summary.json").read_text())
    old=json.loads((ROOT/"results/e262_summary.json").read_text())
    assert len(reports)==len(inventory)==103 and sum(r["available"] for r in inventory)==99
    assert [(r["season"],r["round"]) for r in reports]==[(r["season"],r["round"]) for r in inventory]
    assert all(2018<=r["season"]<=2022 for r in reports)
    counters=Counter();seen_denial=False
    for r,prior in zip(reports,inventory):
        if not prior["available"]:
            assert r["reason"]=="unavailable_in_frozen_car_inventory" and not r["requests"] and not r["available"]
            counters["car_unavailable"]+=1
        elif r["requests"]:
            assert not seen_denial and not r["available"] and len(r["requests"])==1
            assert (r["season"],r["round"])==(2018,1)
            with patch("requests.adapters.HTTPAdapter.send",side_effect=RuntimeError("audit forbids ALL HTTP")):
                s=cached_session(fastf1,2018,1)
            expected="https://livetiming.formula1.com"+s.api_path+"Position.z.jsonStream"
            assert r["requests"]==[dict(url=expected,status=403)]
            seen_denial=True;counters["denied"]+=1
        elif r["available"]:
            assert r["from_cache"] and (r["season"],r["round"]) in ((2019,5),(2019,12),(2019,14))
            original=next(o for o in old["sessions"] if (o["season"],o["round"])==(r["season"],r["round"]))
            with patch("requests.adapters.HTTPAdapter.send",side_effect=RuntimeError("audit forbids ALL HTTP")):
                s=cached_session(fastf1,r["season"],r["round"])
                data=fastf1._api.position_data(s.api_path)
                raw=fastf1._api.fetch_page(s.api_path,"position")
            assert data and hashlib.sha256("".join(line+"\r\n" for line in raw).encode()).hexdigest()==original["position_stream_sha256"]
            counters["cached_available"]+=1
        else:
            assert seen_denial and r["reason"]=="network_unattempted_after_access_denial" and not r["requests"]
            counters["not_attempted"]+=1
        if seen_denial:assert r["stop_remaining"] and r["stop_reason"]=="access_denial"
    assert dict(counters)==dict(denied=1,car_unavailable=4,not_attempted=95,cached_available=3)
    assert result==dict(events=103,available=3,positions_gate=False,actual_requests=1,winner_fit=False)
    assert not (ROOT/"results/e264_inner_predictions.csv").exists()
    print("E264 all103scope/one403/no-later-requests/3cached-hashes/95unknown/coverage/no-winner-fit audit PASS")


if __name__=="__main__":main()
