"""E144: feasibility of a past-race close first-stop undercut label."""

import warnings

import fastf1
import pandas as pd

import f1.ff1  # initialize the existing offline cache
from f1.data import DATA, load_results


PART = DATA / "parts/first_stop_undercuts_2018_2025.csv"


def opportunities(laps, teams):
    l = laps.copy()
    l["lap"] = l.LapNumber.astype(int)
    l = l.set_index(["Driver", "lap"], verify_integrity=True).sort_index()
    first = l[l.PitInTime.notna()].reset_index().groupby("Driver").lap.min()
    rows = []
    for attacker, stop in first.items():
        before = stop - 1
        if (attacker, before) not in l.index:
            continue
        a = l.loc[(attacker, before)]
        if not (2 <= a.Position <= 8 and pd.notna(a.Time)
                and a.TrackStatus == "1" and a.IsAccurate
                and pd.isna(a.PitInTime) and pd.isna(a.PitOutTime)):
            continue
        front = l[(l.index.get_level_values("lap") == before)
                  & (l.Position == a.Position - 1)]
        if len(front) != 1:
            continue
        defender = front.index[0][0]
        b = front.iloc[0]
        gap = (a.Time - b.Time).total_seconds() if pd.notna(b.Time) else float("nan")
        if not (0 <= gap <= 2 and b.TrackStatus == "1" and b.IsAccurate
                and pd.isna(b.PitInTime) and pd.isna(b.PitOutTime)):
            continue
        later = first.get(defender)
        if pd.isna(later) or not 1 <= later - stop <= 5:
            continue
        after = int(later + 2)
        if (attacker, after) not in l.index or (defender, after) not in l.index:
            continue
        end_a, end_b = l.loc[(attacker, after)], l.loc[(defender, after)]
        if any(pd.isna(x.Position) or x.TrackStatus != "1" or not x.IsAccurate
               or pd.notna(x.PitInTime) or pd.notna(x.PitOutTime) for x in (end_a, end_b)):
            continue
        lap_index = l.index.get_level_values("lap")
        window = l[l.index.get_level_values("Driver").isin([attacker, defender])
                   & (lap_index >= before) & (lap_index <= after)]
        if len(window) != 2 * (after - before + 1) or not window.TrackStatus.eq("1").all():
            continue
        rows.append(dict(attacker=attacker, defender=defender, team=teams.get(attacker),
                         before_lap=before, first_stop=stop, later_stop=int(later),
                         after_lap=after, before_position=int(a.Position),
                         gap_seconds=gap, converted=int(end_a.Position < end_b.Position)))
    return pd.DataFrame(rows)


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    fastf1.Cache.offline_mode(True)
    results = load_results().query("season <= 2025")
    rows = []
    for (year, round_), race in results.groupby(["season", "round"], sort=True):
        try:
            session = fastf1.get_session(int(year), int(round_), "R")
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            teams = race.set_index("code").team.to_dict()
            p = opportunities(session.laps, teams)
            if len(p):
                rows.extend(p.assign(season=year, round=round_).to_dict("records"))
            print(year, round_, "opportunities", len(p), flush=True)
        except Exception as exc:
            print(year, round_, "unavailable", type(exc).__name__, str(exc)[:100], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(PART, index=False)
    print("source races", out.groupby(["season", "round"]).ngroups, "opportunities", len(out),
          "conversions", int(out.converted.sum()), "base rate", out.converted.mean())
    print("by year", out.groupby("season").converted.agg(["count", "sum"]).to_string())
    dev = out[out.season.between(2020, 2025)]
    counts = dev.team.value_counts()
    print("development opportunities", len(dev), "teams with >=10", int((counts >= 10).sum()))
    print("examples", out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
