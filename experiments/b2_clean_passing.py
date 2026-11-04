"""E152: clean close-car passing opportunities and past-only driver conversion rate."""

import warnings

import fastf1
import numpy as np
import pandas as pd

import f1.ff1  # initialize the existing offline FastF1 cache
import f1.walkforward as W
from f1.data import DATA, load_results


PART = DATA / "parts/pass_opportunities_2018_2025.csv"


def opportunities(laps):
    cols = ["Driver", "LapNumber", "Position", "Time", "TrackStatus", "IsAccurate", "PitInTime", "PitOutTime"]
    l = laps[cols].copy()
    l = l[l.LapNumber.notna() & l.Position.notna() & l.Time.notna()
          & l.TrackStatus.astype(str).eq("1") & l.IsAccurate.fillna(False)
          & l.PitInTime.isna() & l.PitOutTime.isna()].copy()
    l["lap"] = l.LapNumber.astype(int)
    l["position"] = l.Position.astype(int)
    assert not l.duplicated(["Driver", "lap"]).any()
    current = l[["Driver", "lap", "position", "Time"]]
    following = current[["Driver", "lap", "position"]].copy()
    following["lap"] -= 1
    following = following.rename(columns={"position": "next_position"})
    paired = current.merge(following, on=["Driver", "lap"], validate="one_to_one")
    paired = paired[paired.position.between(1, 8)].copy()
    paired = paired[~paired.duplicated(["lap", "position"], keep=False)]
    rear = paired[paired.position.between(2, 8)].copy()
    rear["front_position"] = rear.position - 1
    front = paired[paired.position.between(1, 7)].rename(columns={
        "Driver": "front", "position": "front_position", "Time": "front_time",
        "next_position": "front_next_position"})
    pair = rear.merge(front[["lap", "front_position", "front", "front_time", "front_next_position"]],
                      on=["lap", "front_position"], validate="many_to_one")
    pair["gap_seconds"] = (pair.Time - pair.front_time).dt.total_seconds()
    pair = pair[pair.gap_seconds.between(0, 2)].copy()
    pair["converted"] = (pair.next_position < pair.front_next_position).astype(int)
    return pair.rename(columns={"Driver": "trailing", "position": "before_position"})[
        ["lap", "trailing", "front", "before_position", "gap_seconds", "converted"]]


def extract(results):
    fastf1.Cache.offline_mode(True)
    rows = pd.read_csv(PART).to_dict("records") if PART.exists() else []
    done = {(x["season"], x["round"]) for x in rows}
    for (season, round_), _ in results.groupby(["season", "round"], sort=True):
        if (season, round_) in done:
            continue
        try:
            session = fastf1.get_session(int(season), int(round_), "R")
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            found = opportunities(session.laps)
            if len(found):
                rows.extend(found.assign(season=season, round=round_).to_dict("records"))
                pd.DataFrame(rows).to_csv(PART, index=False)
            done.add((season, round_))
            print(season, round_, "opportunities", len(found), "conversions", int(found.converted.sum()), flush=True)
        except Exception as exc:
            print(season, round_, "unavailable", type(exc).__name__, str(exc)[:100], flush=True)
    return pd.DataFrame(rows)


def forecast(results, source):
    events = results[["season", "round", "date"]].drop_duplicates().sort_values(["date", "season", "round"])
    events["k"] = np.arange(len(events))
    s = source.merge(events[["season", "round", "k"]], on=["season", "round"], validate="many_to_one")
    output = []
    for k, target in s.groupby("k", sort=True):
        past = s[s.k < k].copy()
        past["w"] = 0.5 ** ((k - past.k) / 50)
        global_rate = (past.w * past.converted).sum() / past.w.sum() if len(past) else 0
        numer = (past.w * past.converted).groupby(past.trailing).sum()
        denom = past.w.groupby(past.trailing).sum()
        d = (numer + 200 * global_rate) / (denom + 200)
        output.append(target[["season", "round", "lap", "trailing", "front", "converted"]].assign(
            driver_prob=target.trailing.map(d).fillna(global_rate).values,
            global_prob=global_rate))
    return pd.concat(output, ignore_index=True)


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    results = load_results().query("season <= 2025")
    source = extract(results)
    assert not source.duplicated(["season", "round", "lap", "trailing"]).any()
    dev = source[source.season.between(2020, 2025)]
    print("source races", source.groupby(["season", "round"]).ngroups,
          "opportunities", len(source), "conversions", int(source.converted.sum()))
    print("per season", source.groupby("season").converted.agg(["count", "sum"]).to_string())
    driver_passes = dev.groupby("trailing").converted.sum()
    outer = dev[dev.season >= 2023]
    gate = (len(dev) >= 10000 and dev.converted.sum() >= 150
            and (driver_passes >= 10).sum() >= 8 and outer.converted.sum() >= 75)
    print("gate", gate, "development opportunities", len(dev), "conversions", int(dev.converted.sum()),
          "drivers with >=10 conversions", int((driver_passes >= 10).sum()),
          "outer conversions", int(outer.converted.sum()))
    if not gate:
        return
    f = forecast(results, source)
    f = f[f.season.between(2020, 2025)].copy()
    for name in ("driver", "global"):
        p = f[name + "_prob"].clip(1e-6, 1 - 1e-6)
        f[name + "_brier"] = (p - f.converted) ** 2
        f[name + "_logloss"] = -(f.converted * np.log(p) + (1 - f.converted) * np.log1p(-p))
    print("per-year", f.groupby("season")[["driver_brier", "global_brier", "driver_logloss", "global_logloss"]].mean().to_string())
    for label, years in (("inner", range(2020, 2023)), ("outer", range(2023, 2026))):
        g = f[f.season.isin(years)]
        race = g.groupby(["season", "round"])[["driver_brier", "global_brier", "driver_logloss", "global_logloss"]].mean()
        print(label, "opportunities", len(g), "races", len(race), "means",
              g[["driver_brier", "global_brier", "driver_logloss", "global_logloss"]].mean().to_dict())
        for metric in ("brier", "logloss"):
            print(label, metric, "driver minus global race-paired",
                  W.paired_boot(race["driver_" + metric], race["global_" + metric])[:2])
        rng = np.random.default_rng(152)
        actual = g.converted.to_numpy()
        predicted = g.driver_prob.to_numpy()
        baseline = g.global_prob.to_numpy()
        groups = list(g.reset_index(drop=True).groupby(["season", "round"]).indices.values())
        observed = ((predicted - actual) ** 2 - (baseline - actual) ** 2).mean()
        null = np.empty(2000)
        for i in range(len(null)):
            shuffled = predicted.copy()
            for ix in groups:
                shuffled[ix] = rng.permutation(shuffled[ix])
            null[i] = ((shuffled - actual) ** 2 - (baseline - actual) ** 2).mean()
        print(label, "within-race shuffled-driver delta Brier null mean", null.mean(),
              "one-sided p", (1 + (null <= observed).sum()) / (len(null) + 1))


if __name__ == "__main__":
    main()
