"""A2 pilot: prior-race clean-air, same-lap pace from cached FastF1 timing."""
import warnings

import fastf1
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import f1.walkforward as W
from experiments.common import table
from f1.data import DATA, load_results
from f1.features import pre_weekend_cols
from f1.ff1 import race_pace

PART = DATA / "parts/matchedpace_2018_2025.csv"


def matched_pace(laps):
    """Same-lap residual pace on green, accurate, clean-air laps; returns gap %."""
    l = laps[(laps.LapNumber > 1) & laps.LapTime.notna() & laps.PitInTime.isna()
             & laps.PitOutTime.isna() & laps.TrackStatus.astype(str).eq("1")
             & laps.IsAccurate.fillna(False)].copy()
    l["t"] = l.LapTime.dt.total_seconds()
    l = l[l.t < 1.07 * l.groupby("Driver").t.transform("median")]
    l = l.sort_values(["LapNumber", "Position"])
    l["ahead"] = l.groupby("LapNumber").Time.shift(1)
    gap = (l.Time - l.ahead).dt.total_seconds()
    l = l[(l.Position == 1) | (gap > 1.5)]
    l = l[l.groupby("LapNumber").Driver.transform("nunique") >= 5]
    l["residual"] = l.t - l.groupby("LapNumber").t.transform("median")
    g = l.groupby("Driver").residual
    p = g.median().where(g.size() >= 8)
    if p.dropna().empty:
        return p.rename("matched_gap")
    return ((p - p.min()) / l.t.median() * 100).rename("matched_gap")


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
            m = matched_pace(session.laps)
            b = race_pace(session.laps)
            frame = pd.concat([m, b], axis=1).reset_index().rename(columns={"Driver": "code"})
            rows.extend(frame.assign(season=year, round=round_).to_dict("records"))
            pd.DataFrame(rows).to_csv(PART, index=False)
            done.add((year, round_))
            print(year, round_, "cached", len(frame), "matched", int(m.notna().sum()), flush=True)
        except Exception as exc:
            print(year, round_, "unavailable", type(exc).__name__, str(exc)[:100], flush=True)
    return pd.DataFrame(rows)


def prior_forecasts(X, pace):
    """For each target race, forecast from strictly earlier race pace rows only."""
    events = X.drop_duplicates(["season", "round"])[["season", "round", "k"]]
    p = pace.merge(events, on=["season", "round"], validate="many_to_one")
    rows = []
    for k, cur in X.groupby("k", sort=True):
        past = p[p.k < k].copy()
        f = cur[["season", "round", "code"]].copy()
        if len(past):
            past["w"] = 0.5 ** ((k - past.k) / 10)
            for source, dest in (("matched_gap", "matched_forecast"), ("pace_gap", "median_forecast")):
                valid = past[past[source].notna()]
                g = (valid[source] * valid.w).groupby(valid.code).sum() / valid.w.groupby(valid.code).sum()
                f[dest] = f.code.map(g)
        else:
            f["matched_forecast"] = f["median_forecast"] = np.nan
        rows.append(f)
    return X.merge(pd.concat(rows, ignore_index=True), on=["season", "round", "code"], validate="one_to_one")


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    results = load_results(); results = results[results.season <= 2025]
    pace = extract(results)
    X = table(); X = X[X.season <= 2025].copy()
    X = prior_forecasts(X, pace)
    base = pre_weekend_cols(X)
    W.assert_pre_weekend(base + ["matched_forecast", "median_forecast"])
    W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
    W.MODELS["logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, base, C=0.03, hl=100)
    W.MODELS["median_pace"] = lambda tr, te, F, **kw: te.median_forecast.fillna(te.median_forecast.max() + 1).values
    W.MODELS["matched_pace"] = lambda tr, te, F, **kw: te.matched_forecast.fillna(te.matched_forecast.max() + 1).values
    def shuffled(tr, te, F, **kw):
        t = tr.copy()
        t["podium"] = np.random.default_rng(0).permutation(t.podium.values)
        return W.m_logit_pod(t, te, base, C=0.03, hl=100)
    W.MODELS["null_shuffled_labels"] = shuffled
    R, P = W.walk_forward(X, [2020, 2021, 2022, 2023, 2024, 2025],
                          ["standings", "logit", "median_pace", "matched_pace", "null_shuffled_labels"],
                          ens={"ens": ["standings", "logit"],
                               "ens_plus_matched": ["standings", "logit", "matched_pace"]}, tie=None)
    R["split"] = np.where(R.season <= 2022, "inner", "outer")
    print("coverage by year", pace.groupby("season").agg(races=("round", "nunique"), rows=("code", "size"), matched=("matched_gap", "count")).to_string())
    print("strict overlap", R.groupby(["model", "split"]).podium.mean().unstack().round(4).to_string())
    picks = P[P.model.isin(["ens", "ens_plus_matched"])].sort_values(["model", "season", "round", "score", "driver"]).groupby(["model", "season", "round"]).head(3)
    picks = picks.groupby(["model", "season", "round"]).driver.apply(frozenset)
    for split, years in (("inner", [2020, 2021, 2022]), ("outer", [2023, 2024, 2025])):
        d = R[R.season.isin(years)].pivot(index=["season", "round"], columns="model", values="podium")
        print(split, "matched - standings", W.paired_boot(d.matched_pace, d.standings)[:2],
              "matched - median pace", W.paired_boot(d.matched_pace, d.median_pace)[:2])
        changed = sum(picks["ens", year, round_] != picks["ens_plus_matched", year, round_]
                      for year, round_ in d.index)
        print(split, "ensemble+matched changes", changed, "of", len(d),
              "vs ens", W.paired_boot(d.ens_plus_matched, d.ens)[:2],
              "vs standings", W.paired_boot(d.ens_plus_matched, d.standings)[:2])
        z = X[X.season.isin(years)].dropna(subset=["y_pace"])
        for name in ("median_forecast", "matched_forecast"):
            corr = z.dropna(subset=[name]).groupby(["season", "round"]).apply(
                lambda g: spearmanr(g[name], g.y_pace)[0] if len(g) >= 6 else np.nan, include_groups=False).mean()
            print(split, name, "race pace rho", round(corr, 4), "coverage", round(z[name].notna().mean(), 3))


if __name__ == "__main__":
    main()
