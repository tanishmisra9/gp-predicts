"""Independent scalar E266 reductions from every cached Q session."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import fastf1
import numpy as np
import pandas as pd
from fastf1.internals.f1account import LivetimingAuthAdapter

import f1.ff1
from f1.data import ROOT, load_results


def scalar(phase):
    values = {}
    if phase is None:
        return values
    seen = set()
    for r in phase.itertuples():
        key = (r.Driver, r.LapNumber)
        assert key not in seen; seen.add(key)
        ts = [r.LapTime, r.Sector1Time, r.Sector2Time, r.Sector3Time]
        if any(pd.isna(t) or t <= pd.Timedelta(0) for t in ts):
            continue
        if abs(sum(ts[1:], pd.Timedelta(0)) - ts[0]) > pd.Timedelta(milliseconds=10):
            continue
        if pd.notna(r.PitInTime) or pd.notna(r.PitOutTime) or str(r.TrackStatus) != "1":
            continue
        if pd.isna(r.IsAccurate) or not r.IsAccurate or r.Compound not in {
                "HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD", "SUPERHARD"}:
            continue
        deleted = getattr(r, "Deleted", None)
        if pd.notna(deleted) and deleted:
            continue
        values.setdefault(r.Driver, {}).setdefault(r.Compound, []).append(ts[0].total_seconds())
    return values


def main():
    reports = json.loads((ROOT / "results/e266_source_status.json").read_text())
    summary = json.loads((ROOT / "results/e266_source_summary.json").read_text())
    source = pd.read_csv(ROOT / "results/e266_observations.csv", float_precision="round_trip")
    entries = load_results().loc[lambda d: d.season.between(2018, 2022), ["season", "round", "code", "team"]]
    assert len(reports) == 103
    pd.testing.assert_frame_equal(source[entries.columns], entries.reset_index(drop=True), check_dtype=False)
    assert [(r["season"], r["round"]) for r in reports] == list(entries[["season", "round"]].drop_duplicates().itertuples(index=False, name=None))
    fastf1.Cache.offline_mode(True)
    supported = available = 0
    for receipt in reports:
        with patch("requests.adapters.HTTPAdapter.send", side_effect=RuntimeError("audit ALL HTTP forbidden")), \
                patch.object(LivetimingAuthAdapter, "_get_token", return_value=None):
            s = fastf1.get_session(receipt["season"], receipt["round"], "Q")
            s.load(laps=True, telemetry=False, weather=False, messages=False)
            q1, _, q3 = s.laps.split_qualifying_sessions()
        assert receipt["q1_present"] == (q1 is not None) and receipt["q3_present"] == (q3 is not None)
        assert receipt["available"] == (q1 is not None and q3 is not None)
        cache = Path(fastf1.Cache._get_cache_file_path(s.api_path, "_extended_timing_data"))
        assert hashlib.sha256(cache.read_bytes()).hexdigest() == receipt["timing_cache_sha256"]
        a, b = scalar(q1), scalar(q3); rows = []
        for driver in sorted(a.keys() & b.keys()):
            common = a[driver].keys() & b[driver].keys()
            if not common:
                continue
            compound = sorted(common, key=lambda c: (min(b[driver][c]), c))[0]
            start, end = min(a[driver][compound]), min(b[driver][compound])
            rows.append(dict(code=driver, compound=compound, q1_seconds=start, q3_seconds=end,
                             q1_laps=len(a[driver][compound]), q3_laps=len(b[driver][compound]),
                             delta_pct=100*(end-start)/start))
        assert receipt["paired_drivers"] == len(rows)
        available += receipt["available"]; supported += len(rows) >= 6
        deltas = [r["delta_pct"] for r in rows]; starts = [r["q1_seconds"] for r in rows]
        event = source[source.season.eq(receipt["season"]) & source["round"].eq(receipt["round"])].set_index("code")
        assert set(event[event.delta_pct.notna()].index) == {r["code"] for r in rows}
        unknown = event.loc[~event.index.isin([r["code"] for r in rows])]
        assert unknown.drop(columns=["season", "round", "team"]).isna().all().all()
        recorded = {r["code"]: r for r in receipt["rows"]}
        assert len(recorded) == len(receipt["rows"]) == len(rows) and set(recorded) == {r["code"] for r in rows}
        for r in rows:
            saved = event.loc[r["code"]]
            assert saved.compound == r["compound"] == recorded[r["code"]]["compound"]
            for c in ("q1_seconds", "q3_seconds", "q1_laps", "q3_laps", "delta_pct"):
                np.testing.assert_allclose(saved[c], r[c], rtol=1e-13, atol=1e-13)
            for c, value in [("evolution_relative", r["delta_pct"]-np.median(deltas) if len(rows)>=6 else np.nan),
                             ("matched_q1_relative", 100*(r["q1_seconds"]-np.median(starts))/min(starts) if len(rows)>=6 else np.nan)]:
                np.testing.assert_allclose(saved[c], value, rtol=1e-13, atol=1e-13, equal_nan=True)
            for c in ["q1_seconds", "q3_seconds", "q1_laps", "q3_laps", "delta_pct", "evolution_relative", "matched_q1_relative"]:
                np.testing.assert_allclose(saved[c], recorded[r["code"]][c], rtol=1e-13, atol=1e-13, equal_nan=True)
        if s._session_split_times:
            splits = s._session_split_times + [s.session_status.Time.iloc[-1]]
            for phase, index in [(q1, 0), (q3, 2)]:
                if phase is not None:
                    nonpit = phase[phase.PitInTime.isna() & phase.PitOutTime.isna()]
                    assert nonpit.LapStartTime.gt(splits[index]).all() and nonpit.LapStartTime.lt(splits[index+1]).all()
    events = entries[["season", "round"]].drop_duplicates().groupby("season").size()
    coverage = source.evolution_relative.notna().groupby(source.season).sum() / (10*events)
    assert summary["available"] == available and summary["supported"] == supported
    for year, value in coverage.items():
        assert np.isclose(summary["year_coverage"][str(year)], value)
    assert summary["source_gate"] == bool(available>=95 and supported>=80 and coverage.ge(.6).all())
    assert not summary["winner_fit"] and not summary["temporal_measured"]
    assert hashlib.sha256((ROOT / "results/e266_observations.csv").read_bytes()).hexdigest() == summary["observations_sha256"]
    print("E266 ALL103cache hashes/scalar phase minima/compound/centering/coverage/phase boundaries audit PASS")


if __name__ == "__main__":
    main()
