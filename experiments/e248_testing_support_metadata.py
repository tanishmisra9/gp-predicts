"""E248 cached testing schedule support audit; metadata-only and post-inspection."""
import hashlib
import json
from unittest.mock import patch

import pandas as pd

from f1.data import ROOT


def summarize(rows):
    assert len({(r["season"],r["test"],r["day"]) for r in rows})==len(rows)
    assert all(r["season"] in (2020,2021,2022) and type(r["api_supported"]) is bool for r in rows)
    by_year={str(y):dict(days=sum(r["season"]==y for r in rows),supported=sum(r["season"]==y and r["api_supported"] for r in rows)) for y in (2020,2021,2022)}
    return dict(by_year=by_year,all_years_have_supported_day=all(v["supported"]>0 for v in by_year.values()),
                missing_2020_2021_supported_days=sum(r["api_supported"] for r in rows if r["season"]<2022),
                raw_timing_acquired=False,raw_endpoint_absence_proven=False,winner_fit=False,
                post_inspection_audit=True)


def main():
    import fastf1
    import fastf1.events
    import f1.ff1
    fastf1.Cache.offline_mode(True);rows=[];receipts=[]
    for year in (2020,2021,2022):
        with patch("requests.adapters.HTTPAdapter.send",side_effect=RuntimeError("metadata audit HTTP forbidden")):
            schedule=fastf1.get_event_schedule(year,include_testing=True,backend="fastf1")
            url=fastf1.events._SCHEDULE_BASE_URL+f"schedule_{year}.json"
            response=fastf1.Cache.requests_get(url,headers=fastf1.events._HEADERS)
        assert response.status_code==200 and response.from_cache
        receipts.append(dict(season=year,url=url,sha256=hashlib.sha256(response.content).hexdigest(),from_cache=True))
        tests=schedule[schedule.RoundNumber.eq(0)]
        for test,(_,event) in enumerate(tests.iterrows(),1):
            assert event.EventFormat=="testing"
            for day in (1,2,3):
                start=event[f"Session{day}DateUtc"]
                assert pd.notna(start) and start.year==year
                rows.append(dict(season=year,test=test,day=day,api_supported=bool(event.F1ApiSupport),start_utc=start.isoformat()))
    report=summarize(rows);report.update(sessions=rows,receipts=receipts)
    (ROOT/"results/e248_testing_support_metadata.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("sessions","receipts")},indent=2))


if __name__=="__main__":main()
