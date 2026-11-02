"""B2 diagnostic: past clean, close-lap pass conversion from cached race timing."""

import warnings

import fastf1
import numpy as np
import pandas as pd

from f1.data import DATA, load_results
import f1.ff1  # initialize the project's existing FastF1 cache
import f1.walkforward as W
from experiments.common import table
from f1.features import pre_weekend_cols


PART = DATA / "parts/passing_2018_2025.csv"


def pass_opportunities(laps):
    """Adjacent, close pairs at one lap line; both must continue cleanly to the next."""
    l = laps[(laps.LapNumber > 1) & laps.Time.notna() & laps.Position.notna()
             & laps.PitInTime.isna() & laps.PitOutTime.isna()
             & laps.TrackStatus.astype(str).eq("1") & laps.IsAccurate.fillna(False)].copy()
    l["lap"] = l.LapNumber.astype(int)
    l["position"] = l.Position.astype(int)
    l = l.sort_values(["lap", "position"])
    l["ahead"] = l.groupby("lap").Driver.shift(1)
    l["ahead_time"] = l.groupby("lap").Time.shift(1)
    l["ahead_position"] = l.groupby("lap").position.shift(1)
    l["gap"] = (l.Time - l.ahead_time).dt.total_seconds()
    l = l[(l.position - l.ahead_position == 1) & l.gap.between(0, 1.5)].copy()
    next_pos = laps[(laps.LapNumber > 2) & laps.Position.notna()
                    & laps.PitInTime.isna() & laps.PitOutTime.isna()
                    & laps.TrackStatus.astype(str).eq("1") & laps.IsAccurate.fillna(False)]
    next_pos = next_pos.assign(lap=next_pos.LapNumber.astype(int) - 1)
    next_pos = next_pos[["lap", "Driver", "Position"]].rename(columns={"Position": "next_position"})
    l = l.merge(next_pos, on=["lap", "Driver"], how="inner", validate="many_to_one")
    l = l.merge(next_pos.rename(columns={"Driver": "ahead", "next_position": "ahead_next_position"}),
                on=["lap", "ahead"], how="inner", validate="many_to_one")
    l["passed"] = (l.next_position < l.ahead_next_position).astype(int)
    return l[["lap", "Driver", "ahead", "gap", "passed"]]


def extract(results):
    fastf1.Cache.offline_mode(True)
    rows = pd.read_csv(PART).to_dict("records") if PART.exists() else []
    done = {(x["season"], x["round"]) for x in rows}
    for (year, round_), _ in results.groupby(["season", "round"], sort=True):
        if (year, round_) in done:
            continue
        try:
            session = fastf1.get_session(int(year), int(round_), "R")
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            p = pass_opportunities(session.laps)
            p = p.assign(season=year, round=round_)
            rows.extend(p.to_dict("records"))
            pd.DataFrame(rows).to_csv(PART, index=False)
            done.add((year, round_))
            print(year, round_, "opportunities", len(p), "passes", int(p.passed.sum()), flush=True)
        except Exception as exc:
            print(year, round_, "unavailable", type(exc).__name__, str(exc)[:100], flush=True)
    return pd.DataFrame(rows)


def prior_rate(events, races):
    """Beta-smoothed driver conversion using strictly earlier events."""
    e = events.merge(races[["season", "round", "k"]].drop_duplicates(),
                     on=["season", "round"], validate="many_to_one")
    rows = []
    for k, cur in races.groupby("k", sort=True):
        past = e[e.k < k]
        g = past.groupby("Driver").passed.agg(["sum", "count"])
        rate = (g["sum"] + 2) / (g["count"] + 20)  # prior mean 0.1
        f = cur[["season", "round", "code", "k"]].copy()
        f["pass_rate"] = f.code.map(rate).fillna(0.1)
        f["pass_n"] = f.code.map(g["count"]).fillna(0)
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    results = load_results()
    results = results[results.season <= 2025]
    e = extract(results)
    print("events", len(e), "passes", int(e.passed.sum()), "races", e.groupby(["season", "round"]).ngroups)
    print("per year", e.groupby("season").passed.agg(["count", "sum"]).to_string())
    print("Bahrain 2023 examples", e[(e.season == 2023) & (e["round"] == 1) & e.passed.eq(1)].head(15).to_string(index=False))
    races = results[["season", "round", "code"]].drop_duplicates().copy()
    races["k"] = races.groupby(["season", "round"], sort=True).ngroup()
    forecast = prior_rate(e, races)
    scored = e.merge(forecast.rename(columns={"code": "Driver"}),
                     on=["season", "round", "Driver"], validate="many_to_one")
    source = e.merge(races[["season", "round", "k"]].drop_duplicates(),
                     on=["season", "round"], validate="many_to_one")
    global_rate = {}
    for k in races.k.unique():
        past = source[source.k < k]
        global_rate[k] = (past.passed.sum() + 2) / (len(past) + 20)
    scored["global_rate"] = scored.k.map(global_rate)
    for name in ("pass_rate", "global_rate"):
        scored[name + "_brier"] = (scored[name] - scored.passed) ** 2
        p = scored[name].clip(1e-6, 1 - 1e-6)
        scored[name + "_logloss"] = -(scored.passed * np.log(p) + (1 - scored.passed) * np.log1p(-p))
    per_race = scored.groupby(["season", "round"])[["pass_rate_brier", "global_rate_brier",
                                                      "pass_rate_logloss", "global_rate_logloss"]].mean().reset_index()
    for split, years in (("inner", range(2020, 2023)), ("outer", range(2023, 2026))):
        z = per_race[per_race.season.isin(years)]
        n = scored[scored.season.isin(years)]
        print(split, "races", len(z), "opportunities", len(n), "pass rate", round(n.passed.mean(), 4),
              "brier driver/global", round(z.pass_rate_brier.mean(), 6), round(z.global_rate_brier.mean(), 6),
              "diff CI", W.paired_boot(z.pass_rate_brier, z.global_rate_brier)[:2],
              "logloss driver/global", round(z.pass_rate_logloss.mean(), 6), round(z.global_rate_logloss.mean(), 6))
    X = table()
    X = X[X.season <= 2025].copy()
    base = pre_weekend_cols(X)
    X = X.merge(forecast[["season", "round", "code", "pass_rate", "pass_n"]],
                on=["season", "round", "code"], validate="one_to_one")
    W.assert_pre_weekend(base + ["pass_rate", "pass_n"])
    W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
    W.MODELS["logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, base, C=0.03, hl=100)
    W.MODELS["past_pass"] = lambda tr, te, F, **kw: -te.pass_rate.values
    def shuffled(tr, te, F, **kw):
        t = tr.copy()
        t["podium"] = np.random.default_rng(0).permutation(t.podium.values)
        return W.m_logit_pod(t, te, base, C=0.03, hl=100)
    W.MODELS["null_shuffled_labels"] = shuffled
    R, P = W.walk_forward(X, range(2020, 2026),
                          ["standings", "logit", "past_pass", "null_shuffled_labels"],
                          ens={"ens": ["standings", "logit"]}, tie=None)
    # Fixed 10% pass signal; standings/logit retain equal shares of the rest.
    # Same within-race rank normalization as the frozen ensemble.
    def blend(g):
        wide = g.pivot(index="driver", columns="model", values="score")
        rank = wide.rank(method="average") / len(wide)
        return 0.45 * rank.standings + 0.45 * rank.logit + 0.10 * rank.past_pass
    new_rows = []
    for (season, round_), g in P.groupby(["season", "round"]):
        te = X[(X.season == season) & (X["round"] == round_)].set_index("driver")
        v = blend(g).reindex(te.index)
        new_rows.append(dict(season=season, round=round_, model="ens_plus_pass",
                             **W.metrics(te.reset_index(), v.values, tie=None)))
    R = pd.concat([R, pd.DataFrame(new_rows)], ignore_index=True)
    R["split"] = np.where(R.season <= 2022, "inner", "outer")
    print("strict overlap", R.groupby(["model", "split"]).podium.mean().unstack().round(4).to_string())
    for split in ("inner", "outer"):
        z = R[R.split == split].pivot(index=["season", "round"], columns="model", values="podium")
        print(split, "pass blend vs standings", W.paired_boot(z.ens_plus_pass, z.standings)[:2],
              "vs ens", W.paired_boot(z.ens_plus_pass, z.ens)[:2])


if __name__ == "__main__":
    main()
