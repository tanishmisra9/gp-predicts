"""Independent native-line decoding and bisect status-coverage audit for E267."""
import bisect
import csv
import hashlib
import io
import json
from unittest.mock import patch

import fastf1
import fastf1._api as api
import pandas as pd
from fastf1.internals.f1account import LivetimingAuthAdapter

import f1.ff1
from f1.data import ROOT


def main():
    result = json.loads((ROOT / "results/e267_status_summary.json").read_text())
    projections = json.loads((ROOT / "results/e267_status_projection.json").read_text())
    saved = list(csv.DictReader(io.StringIO((ROOT / "results/e267_lap_status.csv").read_text())))
    by_event = {(r["season"], r["round"]): r["records"] for r in projections}
    assert result["events"] == result["sessions_available"] == result["raw_available"] == 21
    assert [r["round"] for r in result["sessions"]] == list(range(1, 22))
    assert all(r["season"] == 2018 for r in result["sessions"])
    seen = 0; candidates = 0; blank = 0
    fastf1.Cache.offline_mode(True)
    with patch("requests.adapters.HTTPAdapter.send", side_effect=RuntimeError("audit ALL HTTP forbidden")) as sends, \
            patch.object(LivetimingAuthAdapter, "_get_token", return_value=None):
        for receipt in result["sessions"]:
            s = fastf1.get_session(2018, receipt["round"], "Q")
            s.load(laps=True, telemetry=False, weather=False, messages=False)
            expected_url = api.base_url + s.api_path + "TrackStatus.jsonStream"
            assert receipt["url"] == expected_url
            response = fastf1.Cache.requests_get(expected_url, headers=api.headers)
            assert response.status_code == 200 and response.from_cache and receipt["from_cache"]
            assert hashlib.sha256(response.content).hexdigest() == receipt["raw_sha256"]
            records = []
            # Different clock conversion and newline handling from the extractor.
            for line in response.content.decode("utf-8-sig").split("\n"):
                line = line.removesuffix("\r")
                if not line:
                    continue
                t = pd.Timedelta(line[:12]).value
                row = json.loads(line[12:]); assert isinstance(row, dict)
                records.append(dict(time_ns=t, has_status="Status" in row, status=row.get("Status")))
            assert records == by_event[(2018, receipt["round"])]
            assert receipt["records"] == len(records)
            expected = [(r["time_ns"], r["status"] if r["has_status"] else "") for r in records]
            parsed = [(pd.Timedelta(r.Time).value, r.Status) for r in s.track_status.itertuples()]
            assert receipt["sdk_projection_match"] == (parsed == expected)
            updates = [r for r in records if r["has_status"]]
            times = [r["time_ns"] for r in updates]
            states = [str(r["status"]) if str(r["status"]) in {"1", "2", "4", "5", "6", "7"} else None for r in updates]
            first = next((t for t, status in zip(times, states) if status is not None), None)
            assert first == receipt["first_known_ns"]
            phase_keys = {}
            for index, phase in enumerate(s.laps.split_qualifying_sessions(), 1):
                if phase is not None:
                    for r in phase.itertuples():
                        phase_keys[(r.Driver, float(r.LapNumber))] = f"Q{index}"
            rows = [r for r in saved if int(r["round"]) == receipt["round"]]
            assert len(rows) == len(s.laps) == receipt["laps"]
            event_blank = event_candidates = 0
            for output, native in zip(rows, s.laps.itertuples()):
                assert output["code"] == native.Driver and float(output["lap"]) == native.LapNumber
                phase = phase_keys.get((native.Driver, float(native.LapNumber)), "unknown")
                assert output["phase"] == phase and output["sdk_status"] == str(native.TrackStatus)
                a = None if pd.isna(native.LapStartTime) else pd.Timedelta(native.LapStartTime).value
                b = None if pd.isna(native.Time) else pd.Timedelta(native.Time).value
                assert output["start_ns"] == ("" if a is None else str(a))
                assert output["end_ns"] == ("" if b is None else str(b))
                complete = green = False; codes = set()
                if a is not None and b is not None and b > a:
                    start = bisect.bisect_right(times, a)-1
                    stop = bisect.bisect_right(times, b)
                    selected = states[max(start, 0):stop]
                    codes = {v for v in selected if v is not None}
                    complete = start >= 0 and all(v is not None for v in selected)
                    green = complete and codes == {"1"}
                is_blank = str(native.TrackStatus) == ""
                for name, value in [("sdk_blank", is_blank), ("complete", complete), ("native_green", green),
                                    ("clock_match", parsed == expected),
                                    ("starts_before_first", first is None or a is None or a < first)]:
                    assert output[name] == str(value)
                assert output["codes"] == "".join(sorted(codes))
                event_blank += is_blank
                event_candidates += is_blank and green and parsed == expected and phase == "Q1"
            assert receipt["sdk_blank"] == event_blank and receipt["qualified_green_q1"] == event_candidates
            seen += len(rows); blank += event_blank; candidates += event_candidates
        assert sends.call_count == 0
    assert seen == len(saved) and candidates == result["qualified_green_q1"]
    assert not result["winner_fit"] and not result["data_repaired"] and result["actual_http_sends"] == 0
    assert hashlib.sha256((ROOT / "results/e267_lap_status.csv").read_bytes()).hexdigest() == result["lap_status_sha256"]
    for name, value in result["e266_hashes_unchanged"].items():
        assert hashlib.sha256((ROOT / "results" / name).read_bytes()).hexdigest() == value
    print(f"E267 ALL21raw hashes/independent clocks/bisect ALL{seen}laps/{blank}blanks/{candidates}repair candidates/E266unchanged audit PASS")


if __name__ == "__main__":
    main()
