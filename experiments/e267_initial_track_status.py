"""E267 cached TrackStatus provenance only; never infer an initial green flag."""
import hashlib
import json
import re
from unittest.mock import patch

import pandas as pd

from f1.data import ROOT, load_results

KNOWN = {"1", "2", "4", "5", "6", "7"}


def project(body):
    records = []
    for line in body.decode("utf-8-sig").splitlines():
        if not line:
            continue
        clock = line[:12]
        assert re.fullmatch(r"\d{2}:[0-5]\d:[0-5]\d\.\d{3}", clock), "invalid native clock"
        h, m, s = clock[:8].split(":")
        ns = ((int(h)*3600 + int(m)*60 + int(s))*1000 + int(clock[9:]))*1_000_000
        row = json.loads(line[12:])
        assert isinstance(row, dict), "invalid native patch"
        value = row.get("Status")
        assert value is None or isinstance(value, (str, int)) and not isinstance(value, bool)
        assert not records or ns >= records[-1]["time_ns"], "decreasing native clock"
        records.append(dict(time_ns=ns, has_status="Status" in row, status=value))
    return records


def coverage(records, start, end):
    if start is None or end is None or end <= start:
        return dict(complete=False, native_green=False, codes="")
    state = None
    for r in records:
        if r["time_ns"] > start:
            break
        if r["has_status"]:
            state = str(r["status"]) if str(r["status"]) in KNOWN else None
    complete = state is not None
    codes = {state} if state is not None else set()
    for r in records:
        if start < r["time_ns"] <= end and r["has_status"]:
            state = str(r["status"]) if str(r["status"]) in KNOWN else None
            if state is None:
                complete = False
            else:
                codes.add(state)
    return dict(complete=complete, native_green=complete and codes == {"1"}, codes="".join(sorted(codes)))


def ns(value):
    return None if pd.isna(value) else int(pd.Timedelta(value).value)


def main():
    import fastf1
    import fastf1._api as api
    import f1.ff1
    from fastf1.internals.f1account import LivetimingAuthAdapter
    events = load_results().loc[lambda d: d.season.eq(2018), ["season", "round"]].drop_duplicates()
    assert len(events) == 21
    frozen = {name: hashlib.sha256((ROOT / "results" / name).read_bytes()).hexdigest()
              for name in ["e266_source_status.json", "e266_observations.csv", "e266_source_summary.json"]}
    fastf1.Cache.offline_mode(True)
    reports = []; projections = []; laps = []
    with patch("requests.adapters.HTTPAdapter.send", side_effect=RuntimeError("E267 ALL HTTP forbidden")) as sends, \
            patch.object(LivetimingAuthAdapter, "_get_token", return_value=None):
        for season, round_ in events.itertuples(index=False, name=None):
            report = dict(season=int(season), round=int(round_), session_available=False, raw_available=False)
            try:
                session = fastf1.get_session(int(season), int(round_), "Q")
                session.load(laps=True, telemetry=False, weather=False, messages=False)
                phases = session.laps.split_qualifying_sessions()
            except Exception as exc:
                report["session_error_type"] = type(exc).__name__
                reports.append(report); continue
            report["session_available"] = True
            assert session.api_path.startswith("/static/2018/") and session.api_path.endswith("_Qualifying/")
            url = api.base_url + session.api_path + api.pages["track_status"]
            report["url"] = url
            raw = []; match = None; first = None
            try:
                response = fastf1.Cache.requests_get(url, headers=api.headers)
            except Exception as exc:
                report["raw_error_type"] = type(exc).__name__
            else:
                report["cached_status"] = response.status_code
                report["from_cache"] = bool(getattr(response, "from_cache", False))
                if response.status_code == 200 and report["from_cache"]:
                    raw = project(response.content)
                    sdk = [(ns(r.Time), r.Status) for r in session.track_status.itertuples()]
                    expected = [(r["time_ns"], r["status"] if r["has_status"] else "") for r in raw]
                    match = sdk == expected
                    first = next((r["time_ns"] for r in raw if r["has_status"] and str(r["status"]) in KNOWN), None)
                    report.update(raw_available=True, raw_sha256=hashlib.sha256(response.content).hexdigest())
                    projections.append(dict(season=int(season), round=int(round_), records=raw))
            report.update(records=len(raw), sdk_projection_match=match, first_known_ns=first)
            phase_keys = {}
            for index, phase in enumerate(phases, 1):
                if phase is not None:
                    for r in phase.itertuples():
                        key = (r.Driver, r.LapNumber)
                        assert key not in phase_keys
                        phase_keys[key] = f"Q{index}"
            event_rows = []
            for r in session.laps.itertuples():
                start, end = ns(r.LapStartTime), ns(r.Time)
                row = dict(season=int(season), round=int(round_), code=r.Driver, lap=float(r.LapNumber),
                           phase=phase_keys.get((r.Driver, r.LapNumber), "unknown"), start_ns=start, end_ns=end,
                           sdk_status=str(r.TrackStatus), sdk_blank=str(r.TrackStatus) == "",
                           starts_before_first=first is None or start is None or start < first,
                           clock_match=match, **coverage(raw, start, end))
                event_rows.append(row)
            report.update(laps=len(event_rows), sdk_blank=sum(r["sdk_blank"] for r in event_rows),
                          qualified_green_q1=sum(r["sdk_blank"] and r["native_green"] and match and r["phase"] == "Q1"
                                                 for r in event_rows))
            laps.extend(event_rows); reports.append(report)
            print("SOURCE", season, round_, len(raw), match, report["sdk_blank"], report["qualified_green_q1"], flush=True)
        assert sends.call_count == 0, "a native send was attempted"
    for name, digest in frozen.items():
        assert hashlib.sha256((ROOT / "results" / name).read_bytes()).hexdigest() == digest
    table = pd.DataFrame(laps)
    table.to_csv(ROOT / "results/e267_lap_status.csv", index=False)
    result = dict(events=21, sessions_available=sum(r["session_available"] for r in reports),
                  raw_available=sum(r["raw_available"] for r in reports),
                  qualified_green_q1=sum(r.get("qualified_green_q1", 0) for r in reports),
                  actual_http_sends=0, winner_fit=False, data_repaired=False,
                  e266_hashes_unchanged=frozen, sessions=reports,
                  lap_status_sha256=hashlib.sha256((ROOT / "results/e267_lap_status.csv").read_bytes()).hexdigest())
    for name, value in [("e267_status_projection.json", projections), ("e267_status_summary.json", result)]:
        with (ROOT / "results" / name).open("x") as file:
            json.dump(value, file, indent=2); file.write("\n")
    print("RESULT", json.dumps({k: v for k, v in result.items() if k not in {"sessions", "e266_hashes_unchanged"}}), flush=True)


if __name__ == "__main__":
    main()
