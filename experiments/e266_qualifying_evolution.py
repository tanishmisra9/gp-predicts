"""E266 cached, same-compound actual Q1-to-Q3 evolution; no winner fit."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.e208_ideal_q1 import DRY
from f1.data import ROOT, load_results

FIELDS = ["code", "compound", "q1_seconds", "q3_seconds", "q1_laps", "q3_laps",
          "delta_pct", "evolution_relative", "matched_q1_relative"]
TIMES = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]
STATUS = ROOT / "results/e266_source_status.json"


def clean(frame):
    if frame is None:
        return pd.DataFrame(columns=["Driver", "Compound", "seconds"])
    assert not frame.duplicated(["Driver", "LapNumber"]).any()
    data = frame.copy()
    times = np.column_stack([data[c].dt.total_seconds() for c in TIMES])
    good = (data.PitInTime.isna() & data.PitOutTime.isna()
            & data.TrackStatus.astype(str).eq("1") & data.IsAccurate.fillna(False)
            & data.Compound.isin(DRY))
    good &= np.isfinite(times).all(axis=1) & (times > 0).all(axis=1)
    # Use timedeltas so the inclusive10ms boundary is not a float subtraction.
    difference = data[TIMES[1:]].sum(axis=1) - data.LapTime
    good &= difference.abs().le(pd.Timedelta(milliseconds=10))
    if "Deleted" in data:
        good &= ~data.Deleted.fillna(False).astype(bool)
    return data.assign(seconds=times[:, 0]).loc[good]


def measure(q1, q3):
    first, last = clean(q1), clean(q3)
    rows = []
    for driver in sorted(set(first.Driver) & set(last.Driver)):
        a = first[first.Driver.eq(driver)]; b = last[last.Driver.eq(driver)]
        choices = []
        for compound in sorted(set(a.Compound) & set(b.Compound)):
            x = a[a.Compound.eq(compound)].seconds
            y = b[b.Compound.eq(compound)].seconds
            choices.append((float(y.min()), compound, float(x.min()), len(x), len(y)))
        if choices:
            end, compound, start, n1, n3 = min(choices)
            rows.append(dict(code=driver, compound=compound, q1_seconds=start,
                             q3_seconds=end, q1_laps=n1, q3_laps=n3,
                             delta_pct=100 * (end - start) / start))
    out = pd.DataFrame(rows, columns=FIELDS[:7])
    out["evolution_relative"] = out.delta_pct - out.delta_pct.median() if len(out) >= 6 else np.nan
    out["matched_q1_relative"] = (100 * (out.q1_seconds - out.q1_seconds.median())
                                  / out.q1_seconds.min()) if len(out) >= 6 else np.nan
    return out[FIELDS]


def source_summary(reports, table, entries):
    events = entries[["season", "round"]].drop_duplicates().groupby("season").size()
    coverage = table.evolution_relative.notna().groupby(table.season).sum() / (10 * events)
    available = sum(r["available"] for r in reports)
    supported = sum(r.get("paired_drivers", 0) >= 6 for r in reports)
    return dict(events=len(reports), available=available, supported=supported,
                year_coverage=coverage.to_dict(), source_gate=bool(
                    len(reports) == 103 and available >= 95 and supported >= 80
                    and coverage.ge(.60).all()), temporal_measured=False, winner_fit=False)


def extract():
    import fastf1
    import f1.ff1
    from fastf1.internals.f1account import LivetimingAuthAdapter
    entries = load_results().loc[lambda d: d.season.between(2018, 2022),
                                 ["season", "round", "code", "team"]]
    assert not entries.duplicated(["season", "round", "code"]).any()
    events = entries[["season", "round"]].drop_duplicates()
    assert len(events) == 103
    fastf1.Cache.offline_mode(True)
    reports = json.loads(STATUS.read_text()) if STATUS.exists() else []
    done = {(r["season"], r["round"]) for r in reports}
    assert len(done) == len(reports) and done <= set(map(tuple, events.to_numpy()))
    for season, round_ in events.itertuples(index=False, name=None):
        if (season, round_) in done:
            continue
        r = dict(season=int(season), round=int(round_), available=False, rows=[])
        try:
            with patch("requests.adapters.HTTPAdapter.send", side_effect=RuntimeError("E266 ALL HTTP forbidden")), \
                    patch.object(LivetimingAuthAdapter, "_get_token", return_value=None):
                session = fastf1.get_session(int(season), int(round_), "Q")
                session.load(laps=True, telemetry=False, weather=False, messages=False)
                q1, _, q3 = session.laps.split_qualifying_sessions()
        except Exception as exc:
            r["error_type"] = type(exc).__name__
        else:
            r.update(q1_present=q1 is not None, q3_present=q3 is not None)
            measured = measure(q1, q3)
            codes = set(entries.loc[entries.season.eq(season) & entries["round"].eq(round_), "code"])
            assert set(measured.code) <= codes
            r.update(available=q1 is not None and q3 is not None,
                     paired_drivers=len(measured), rows=measured.to_dict("records"))
            # Hash the existing native parsed timing cache, without fetching a page.
            cache = fastf1.Cache._get_cache_file_path(session.api_path, "_extended_timing_data")
            r["timing_cache_sha256"] = hashlib.sha256(Path(cache).read_bytes()).hexdigest()
        reports.append(r)
        temporary = STATUS.with_suffix(".tmp")
        temporary.write_text(json.dumps(reports, indent=2) + "\n"); temporary.replace(STATUS)
        print("SOURCE", season, round_, r["available"], r.get("paired_drivers", 0), flush=True)
    observations = pd.DataFrame([dict(season=r["season"], round=r["round"], **row)
                                 for r in reports for row in r["rows"]],
                                columns=["season", "round"] + FIELDS)
    table = entries.merge(observations, on=["season", "round", "code"], how="left", validate="one_to_one")
    summary = source_summary(reports, table, entries)
    table.to_csv(ROOT / "results/e266_observations.csv", index=False)
    summary["observations_sha256"] = hashlib.sha256((ROOT / "results/e266_observations.csv").read_bytes()).hexdigest()
    (ROOT / "results/e266_source_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("RESULT", json.dumps(summary), flush=True)
    return table, summary


if __name__ == "__main__":
    extract()
