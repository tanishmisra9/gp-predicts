"""E264 fixed2018-22 native position acquisition; no profiles or outcomes."""
import json
import shutil
from unittest.mock import patch
from experiments.e190_traffic_feasibility import cached_session
from experiments.e262_drs_support import bounded_send
from f1.data import ROOT

STATUS=ROOT/"results/e264_position_status.json"
FLOOR=15*1024**3


def allowed_urls(path,season):
    assert 2018<=season<=2022 and path.startswith(f"/static/{season}/") and path.endswith("_Race/")
    return {host+path+"Position.z.jsonStream" for host in ("https://livetiming.formula1.com","https://livetiming-mirror.fastf1.dev")}


def acquire(inventory,status,session_loader,position_loader,cache_mode,authenticate,send_owner,free_bytes):
    assert len(inventory)==103 and sum(r["available"] for r in inventory)==99
    legal=[(r["season"],r["round"]) for r in inventory]
    assert len(set(legal))==103 and all(2018<=year<=2022 for year,_ in legal)
    reports=json.loads(status.read_text()) if status.exists() else []
    assert [(r["season"],r["round"]) for r in reports]==legal[:len(reports)]
    stop={"denied":any(r.get("stop_remaining",False) for r in reports)}
    reason=next((r["stop_reason"] for r in reversed(reports) if r.get("stop_remaining",False)),None)
    authenticated=False
    for receipt in inventory[len(reports):]:
        year,round_=receipt["season"],receipt["round"]
        report=dict(season=year,round=round_,available=False,requests=[])
        if not receipt["available"]:report["reason"]="unavailable_in_frozen_car_inventory"
        else:
            cache_mode(True)
            try:session=session_loader(year,round_)
            except Exception as exc:report["error_type"]=type(exc).__name__
            else:
                assert allowed_urls(session.api_path,year)
                try:
                    with patch("requests.adapters.HTTPAdapter.send",side_effect=RuntimeError("E264 cache probe HTTP forbidden")):
                        positions=position_loader(session.api_path)
                except Exception:
                    if free_bytes()<FLOOR:
                        stop["denied"]=True;reason="disk_floor"
                    if stop["denied"]:report["reason"]="network_unattempted_after_"+str(reason)
                    else:
                        cache_mode(False)
                        if not authenticated:
                            try:authenticate()
                            except Exception as exc:
                                report["error_type"]=type(exc).__name__;stop["denied"]=True;reason="authentication_failure"
                            else:authenticated=True
                        if authenticated:
                            try:
                                with patch.object(send_owner,"send",new=bounded_send(send_owner.send,allowed_urls(session.api_path,year),report["requests"],stop)):
                                    positions=position_loader(session.api_path)
                            except Exception as exc:report["error_type"]=type(exc).__name__
                            else:report["available"]=bool(positions)
                            if any(r.get("status") in (401,403,429) for r in report["requests"]):reason="access_denial"
                else:
                    report["available"]=bool(positions)
                    report["from_cache"]=True
        report.update(stop_remaining=stop["denied"],stop_reason=reason)
        reports.append(report)
        temporary=status.with_suffix(".tmp");temporary.write_text(json.dumps(reports,indent=2)+"\n");temporary.replace(status)
        print("POSITION",year,round_,report["available"],len(report["requests"]),report.get("reason",""),flush=True)
    return reports


def main():
    import fastf1
    import requests
    import f1.ff1
    from fastf1.internals.f1account import get_token
    inventory=json.loads((ROOT/"results/e212_source_status.json").read_text())
    reports=acquire(inventory,STATUS,lambda year,round_:cached_session(fastf1,year,round_),fastf1._api.position_data,
                    fastf1.Cache.offline_mode,lambda:get_token(method="password"),requests.adapters.HTTPAdapter,
                    lambda:shutil.disk_usage(ROOT).free)
    result=dict(events=len(reports),available=sum(r["available"] for r in reports),positions_gate=sum(r["available"] for r in reports)>=95,
                actual_requests=sum(len(r["requests"]) for r in reports),winner_fit=False)
    (ROOT/"results/e264_position_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
