import hashlib
import json
from datetime import datetime,timedelta,timezone
from unittest.mock import patch

from experiments.e248_testing_support_metadata import summarize
from f1.data import ROOT


def main():
    base=[dict(season=y,test=1,day=1,api_supported=False) for y in (2020,2021,2022)]
    assert not summarize(base)["all_years_have_supported_day"]
    assert summarize([dict(r,api_supported=True) for r in base])["all_years_have_supported_day"]
    assert not summarize([])["all_years_have_supported_day"]
    try:summarize(base+base[:1])
    except AssertionError:pass
    else:raise AssertionError("duplicate accepted")
    import fastf1
    import fastf1.events
    import f1.ff1
    fastf1.Cache.offline_mode(True)
    report=json.loads((ROOT/"results/e248_testing_support_metadata.json").read_text())
    rows=[]
    for receipt in report["receipts"]:
        year=receipt["season"]
        assert receipt["url"]==fastf1.events._SCHEDULE_BASE_URL+f"schedule_{year}.json"
        with patch("requests.adapters.HTTPAdapter.send",side_effect=RuntimeError("audit HTTP forbidden")):
            response=fastf1.Cache.requests_get(receipt["url"],headers=fastf1.events._HEADERS)
        assert response.from_cache and response.status_code==200 and hashlib.sha256(response.content).hexdigest()==receipt["sha256"]
        raw=json.loads(response.content)
        indices=[key for key,value in raw["round_number"].items() if value==0]
        for test,key in enumerate(indices,1):
            assert raw["event_format"][key]=="testing" and type(raw["f1_api_support"][key]) is bool
            offset=raw["gmt_offset"][key]
            sign=-1 if offset.startswith("-") else 1
            hours,minutes=map(int,offset.lstrip("+-").split(":"))
            tz=timezone(sign*timedelta(hours=hours,minutes=minutes))
            for day in (1,2,3):
                local=datetime.fromisoformat(raw[f"session{day}_date"][key]).replace(tzinfo=tz)
                utc=local.astimezone(timezone.utc).replace(tzinfo=None)
                rows.append(dict(season=year,test=test,day=day,api_supported=raw["f1_api_support"][key],start_utc=utc.isoformat()))
    assert rows==report["sessions"]
    expected={str(y):dict(days=sum(r["season"]==y for r in rows),supported=sum(r["season"]==y and r["api_supported"] for r in rows)) for y in (2020,2021,2022)}
    assert expected==report["by_year"]
    assert report["all_years_have_supported_day"]==all(v["supported"]>0 for v in expected.values())
    assert report["missing_2020_2021_supported_days"]==sum(r["api_supported"] for r in rows if r["season"]<2022)
    assert report["post_inspection_audit"] and not any(report[k] for k in ("raw_timing_acquired","raw_endpoint_absence_proven","winner_fit"))
    print("E248 synthetic missing/positive/duplicate and independent raw schedule flags/counts/UTC/hash audit PASS")


if __name__=="__main__":main()
