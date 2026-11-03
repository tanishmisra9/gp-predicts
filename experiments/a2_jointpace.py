"""A2: jointly adjust historical race pace for compound, tyre age, and traffic."""

import warnings

import fastf1
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

import f1.ff1  # initialize the project's existing FastF1 cache
import f1.walkforward as W
from experiments.a2_matchedpace import PART as MATCHED_PART, prior_forecasts
from experiments.common import table
from f1.data import DATA, load_results
from f1.features import pre_weekend_cols


PART = DATA / "parts/jointpace_2018_2025.csv"


def joint_pace(laps):
    line = laps[laps.Time.notna() & laps.Position.notna()].sort_values(["LapNumber", "Position"]).copy()
    line["ahead_time"] = line.groupby("LapNumber").Time.shift(1)
    line["gap_ahead"] = (line.Time - line.ahead_time).dt.total_seconds()
    l = line[(line.LapNumber > 1) & line.LapTime.notna() & line.PitInTime.isna()
             & line.PitOutTime.isna() & line.TrackStatus.astype(str).eq("1")
             & line.IsAccurate.fillna(False) & line.TyreLife.notna() & line.Compound.notna()].copy()
    l["t"] = l.LapTime.dt.total_seconds()
    l = l[l.t < 1.07 * l.groupby("Driver").t.transform("median")]
    if len(l) < 100:
        return pd.Series(dtype=float, name="joint_gap")
    l["residual"] = l.t - l.groupby("LapNumber").t.transform("median")
    l["traffic"] = l.gap_ahead.between(0, 1.5).astype(float)
    comp = pd.get_dummies(l.Compound, prefix="compound", dtype=float)
    age = comp.mul((l.TyreLife - l.groupby("Compound").TyreLife.transform("median")) / 10, axis=0)
    age.columns = ["age_" + c for c in comp]
    nuisance = pd.concat([comp, age, l[["traffic"]]], axis=1)
    driver = pd.get_dummies(l.Driver, prefix="driver", dtype=float)
    design = pd.concat([driver, nuisance], axis=1)
    fit = Ridge(alpha=50).fit(design, l.residual)
    corrected = l.residual - nuisance.to_numpy() @ fit.coef_[len(driver.columns):]
    g = corrected.groupby(l.Driver)
    p = g.median().where(g.size() >= 8)
    if p.dropna().empty:
        return p.rename("joint_gap")
    return ((p - p.min()) / l.t.median() * 100).rename("joint_gap")


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
            p = joint_pace(session.laps).reset_index().rename(columns={"Driver": "code"})
            if len(p):
                rows.extend(p.assign(season=year, round=round_).to_dict("records"))
                pd.DataFrame(rows).to_csv(PART, index=False)
            done.add((year, round_))
            print(year, round_, "joint rows", p.joint_gap.notna().sum(), flush=True)
        except Exception as exc:
            print(year, round_, "unavailable", type(exc).__name__, str(exc)[:100], flush=True)
    return pd.DataFrame(rows)


def prior_joint(X, source):
    events = X[["season", "round", "k"]].drop_duplicates()
    p = source.merge(events, on=["season", "round"], validate="many_to_one")
    rows = []
    for k, cur in X.groupby("k", sort=True):
        past = p[(p.k < k) & p.joint_gap.notna()].copy()
        past["w"] = 0.5 ** ((k - past.k) / 10)
        g = (past.joint_gap * past.w).groupby(past.code).sum() / past.w.groupby(past.code).sum()
        f = cur[["season", "round", "code"]].copy()
        f["joint_forecast"] = f.code.map(g)
        rows.append(f)
    return X.merge(pd.concat(rows, ignore_index=True), on=["season", "round", "code"], validate="one_to_one")


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    results = load_results()
    results = results[results.season <= 2025]
    source = extract(results)
    matched = pd.read_csv(MATCHED_PART)
    X = table()
    X = X[X.season <= 2025].copy()
    X = prior_forecasts(X, matched)
    X = prior_joint(X, source)
    base = pre_weekend_cols(X)
    W.assert_pre_weekend(base + ["joint_forecast", "matched_forecast", "median_forecast"])
    W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
    W.MODELS["logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, base, C=0.03, hl=100)
    for name in ("joint", "matched", "median"):
        W.MODELS[name] = lambda tr, te, F, col=name + "_forecast", **kw: te[col].fillna(te[col].max() + 1).values
    def shuffled(tr, te, F, **kw):
        t = tr.copy()
        t["podium"] = np.random.default_rng(0).permutation(t.podium.values)
        return W.m_logit_pod(t, te, base, C=0.03, hl=100)
    W.MODELS["null_shuffled_labels"] = shuffled
    R, P = W.walk_forward(X, range(2020, 2026),
                          ["standings", "logit", "joint", "matched", "median", "null_shuffled_labels"],
                          ens={"ens": ["standings", "logit"],
                               "ens_joint": ["standings", "logit", "joint"]}, tie=None)
    R["split"] = np.where(R.season <= 2022, "inner", "outer")
    print("source coverage", source.groupby("season").agg(races=("round", "nunique"), rows=("code", "size"), valid=("joint_gap", "count")).to_string())
    print("strict overlap", R.groupby(["model", "split"]).podium.mean().unstack().round(4).to_string())
    print("strict overlap by season", R[R.season >= 2023].groupby(["model", "season"]).podium.mean().unstack().round(4).to_string())
    picks = P[P.model.isin(["ens", "ens_joint"])].sort_values(["model", "season", "round", "score", "driver"])
    picks = picks.groupby(["model", "season", "round"]).head(3)
    picks = picks.groupby(["model", "season", "round"]).driver.apply(frozenset)
    for split, years in (("inner", range(2020, 2023)), ("outer", range(2023, 2026))):
        z = R[R.split == split].pivot(index=["season", "round"], columns="model", values="podium")
        changed = sum(picks["ens", year, round_] != picks["ens_joint", year, round_] for year, round_ in z.index)
        print(split, "joint vs standings", W.paired_boot(z.joint, z.standings)[:2],
              "vs matched", W.paired_boot(z.joint, z.matched)[:2],
              "blend vs ens", W.paired_boot(z.ens_joint, z.ens)[:2],
              "vs standings", W.paired_boot(z.ens_joint, z.standings)[:2],
              "top3 changed", changed)
        d = X[X.season.isin(years)].dropna(subset=["y_pace"])
        for col in ("joint_forecast", "matched_forecast", "median_forecast"):
            q = d.dropna(subset=[col]).groupby(["season", "round"]).apply(
                lambda g: spearmanr(g[col], g.y_pace)[0] if len(g) >= 6 else np.nan, include_groups=False).mean()
            print(split, col, "next-race pace rho", round(q, 4), "coverage", round(d[col].notna().mean(), 4))
        common = d.dropna(subset=["joint_forecast", "matched_forecast"])
        rho = common.groupby(["season", "round"]).apply(
            lambda g: pd.Series({col: spearmanr(g[col], g.y_pace)[0] for col in ("joint_forecast", "matched_forecast")})
            if len(g) >= 6 else pd.Series({"joint_forecast": np.nan, "matched_forecast": np.nan}),
            include_groups=False).dropna()
        print(split, "joint-minus-matched pace rho paired", W.paired_boot(rho.joint_forecast, rho.matched_forecast)[:2])


if __name__ == "__main__":
    main()
