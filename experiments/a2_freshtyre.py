"""E119: does used-versus-fresh tyre state improve past race-pace correction?"""

import warnings

import fastf1
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

import f1.ff1  # initialize the project's existing FastF1 cache
import f1.walkforward as W
from experiments.a2_jointpace import PART as JOINT_PART, prior_joint
from experiments.common import table
from f1.data import DATA, load_results
from f1.features import pre_weekend_cols


PART = DATA / "parts/jointpace_fresh_2018_2025.csv"


def fresh_pace(laps):
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
    l["used"] = l.FreshTyre.eq(False).astype(float)
    comp = pd.get_dummies(l.Compound, prefix="compound", dtype=float)
    age = comp.mul((l.TyreLife - l.groupby("Compound").TyreLife.transform("median")) / 10, axis=0)
    age.columns = ["age_" + c for c in comp]
    nuisance = pd.concat([comp, age, l[["traffic", "used"]]], axis=1)
    driver = pd.get_dummies(l.Driver, prefix="driver", dtype=float)
    fit = Ridge(alpha=50).fit(pd.concat([driver, nuisance], axis=1), l.residual)
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
            p = fresh_pace(session.laps).reset_index().rename(columns={"Driver": "code"})
            if len(p):
                rows.extend(p.assign(season=year, round=round_).to_dict("records"))
                pd.DataFrame(rows).to_csv(PART, index=False)
            done.add((year, round_))
            print(year, round_, "fresh rows", p.joint_gap.notna().sum(), flush=True)
        except Exception as exc:
            print(year, round_, "unavailable", type(exc).__name__, str(exc)[:100], flush=True)
    return pd.DataFrame(rows)


def feature_table(X, fresh):
    f = prior_joint(X[["season", "round", "k", "code"]], fresh)[["season", "round", "code", "joint_forecast"]].rename(
        columns={"joint_forecast": "fresh_forecast"})
    return X.merge(f, on=["season", "round", "code"], validate="one_to_one")


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    results = load_results(); results = results[results.season <= 2025]
    fresh = extract(results)
    joint = pd.read_csv(JOINT_PART)
    X = table(); X = X[X.season <= 2025].copy()
    X = prior_joint(X, joint)
    X = feature_table(X, fresh)
    base = pre_weekend_cols(X)
    W.assert_pre_weekend(base + ["joint_forecast", "fresh_forecast"])
    W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
    W.MODELS["logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, base, C=0.03, hl=100)
    for name, col in (("joint", "joint_forecast"), ("fresh", "fresh_forecast")):
        W.MODELS[name] = lambda tr, te, F, col=col, **kw: te[col].fillna(te[col].max() + 1).values

    def shuffled(tr, te, F, **kw):
        t = tr.copy(); t["podium"] = np.random.default_rng(0).permutation(t.podium.values)
        return W.m_logit_pod(t, te, base, C=0.03, hl=100)

    W.MODELS["null_shuffled_labels"] = shuffled
    R, P = W.walk_forward(X, range(2020, 2026),
                          ["standings", "logit", "joint", "fresh", "null_shuffled_labels"],
                          ens={"ens": ["standings", "logit"],
                               "ens_fresh": ["standings", "logit", "fresh"]}, tie=None)
    R["split"] = np.where(R.season <= 2022, "inner", "outer")
    print("coverage", fresh.groupby("season").agg(races=("round", "nunique"), rows=("code", "size"), valid=("joint_gap", "count")).to_string())
    print("strict overlap", R.groupby(["model", "split"]).podium.mean().unstack().round(4).to_string())
    for split, years in (("inner", range(2020, 2023)), ("outer", range(2023, 2026))):
        z = R[R.split == split].pivot(index=["season", "round"], columns="model", values="podium")
        for candidate, reference in (("fresh", "joint"), ("ens_fresh", "ens"), ("ens_fresh", "standings")):
            diff, ci, _ = W.paired_boot(z[candidate], z[reference])
            print(split, "podium", candidate, "vs", reference, round(diff, 4), np.round(ci, 4).tolist())
        d = X[X.season.isin(years)].dropna(subset=["y_pace", "joint_forecast", "fresh_forecast"])
        rho = d.groupby(["season", "round"]).apply(
            lambda g: pd.Series({c: spearmanr(g[c], g.y_pace)[0] for c in ("joint_forecast", "fresh_forecast")})
            if len(g) >= 6 else pd.Series({"joint_forecast": np.nan, "fresh_forecast": np.nan}),
            include_groups=False).dropna()
        diff, ci, _ = W.paired_boot(rho.fresh_forecast, rho.joint_forecast)
        print(split, "pace rho", rho.mean().round(4).to_dict(), "fresh minus joint", round(diff, 4), np.round(ci, 4).tolist(), "races", len(rho))


if __name__ == "__main__":
    main()
