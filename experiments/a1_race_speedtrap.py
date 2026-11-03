"""E126: many-lap race speed-trap advantage as a past-only podium feature."""

import warnings

import fastf1
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import f1.walkforward as W
from experiments.common import table
from f1.data import DATA, load_results
from f1.features import pre_weekend_cols


PART = DATA / "parts/race_speedtrap_2018_2025.csv"


def race_speed(laps):
    """Median green, non-pit, same-lap SpeedST advantage (km/h) per driver."""
    l = laps[(laps.LapNumber > 1) & laps.LapTime.notna() & laps.SpeedST.notna()
             & laps.PitInTime.isna() & laps.PitOutTime.isna()
             & laps.TrackStatus.astype(str).eq("1") & laps.IsAccurate.fillna(False)].copy()
    l["t"] = l.LapTime.dt.total_seconds()
    l = l[l.t < 1.07 * l.groupby("Driver").t.transform("median")].copy()
    l = l[l.groupby("LapNumber").SpeedST.transform("count") >= 6].copy()
    l["resid"] = l.SpeedST - l.groupby("LapNumber").SpeedST.transform("median")
    g = l.groupby("Driver").resid
    return g.median().where(g.size() >= 8).dropna().rename("speed_adv")


def extract(results):
    fastf1.Cache.enable_cache(str(DATA / "fastf1_cache"))
    fastf1.Cache.offline_mode(True)
    rows = pd.read_csv(PART).to_dict("records") if PART.exists() else []
    done = {(x["season"], x["round"]) for x in rows}
    for (season, round_), _ in results.groupby(["season", "round"], sort=True):
        if (season, round_) in done:
            continue
        try:
            session = fastf1.get_session(int(season), int(round_), "R")
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            p = race_speed(session.laps).reset_index().rename(columns={"Driver": "code"})
            if len(p):
                rows.extend(p.assign(season=season, round=round_).to_dict("records"))
                pd.DataFrame(rows).to_csv(PART, index=False)
            done.add((season, round_))
            print(season, round_, "speed rows", len(p), flush=True)
        except Exception as exc:
            print(season, round_, "unavailable", type(exc).__name__, str(exc)[:100], flush=True)
    return pd.DataFrame(rows)


def prior_speed(X, source):
    """Exponential team mean from strictly earlier race summaries."""
    keys = X[["season", "round", "code", "team", "k"]]
    s = source.merge(keys, on=["season", "round", "code"], validate="one_to_one")
    team = s.groupby(["season", "round", "team", "k"], as_index=False).speed_adv.mean()
    out = []
    for k, cur in X.groupby("k", sort=True):
        past = team[team.k < k].copy()
        past["w"] = 0.5 ** ((k - past.k) / 10)
        num = (past.speed_adv * past.w).groupby(past.team).sum()
        den = past.w.groupby(past.team).sum()
        f = cur[["season", "round", "code", "team"]].copy()
        f["prior_speed_adv"] = f.team.map(num / (den + 10))
        out.append(f.drop(columns="team"))
    return X.merge(pd.concat(out, ignore_index=True), on=["season", "round", "code"], validate="one_to_one")


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    results = load_results()
    source = extract(results[results.season <= 2025])
    X = table(); X = X[X.season <= 2025].copy()
    X = prior_speed(X, source)
    base = pre_weekend_cols(X)
    W.assert_pre_weekend(base + ["prior_speed_adv"])
    W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
    W.MODELS["logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, base, C=0.03, hl=100)
    W.MODELS["logit_speed"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, base + ["prior_speed_adv"], C=0.03, hl=100)

    def shuffled(tr, te, F, **kw):
        t = tr.copy(); t["podium"] = np.random.default_rng(0).permutation(t.podium.values)
        return W.m_logit_pod(t, te, base, C=0.03, hl=100)

    W.MODELS["null_shuffled_labels"] = shuffled
    R, P = W.walk_forward(X, range(2020, 2026),
                          ["standings", "logit", "logit_speed", "null_shuffled_labels"],
                          ens={"ens": ["standings", "logit"],
                               "ens_speed": ["standings", "logit_speed"]}, tie=None)
    R["split"] = np.where(R.season <= 2022, "inner", "outer")
    print("source coverage", source.groupby("season").agg(races=("round", "nunique"), rows=("code", "size")).to_string())
    print("feature coverage", X[X.season.between(2020, 2025)].groupby("season").prior_speed_adv.apply(lambda s: s.notna().mean()).round(4).to_string())
    print("strict overlap", R.groupby(["model", "split"]).podium.mean().unstack().round(4).to_string())
    for split, years in (("inner", range(2020, 2023)), ("outer", range(2023, 2026))):
        z = R[R.split == split].pivot(index=["season", "round"], columns="model", values="podium")
        print(split, "speed ensemble vs ens", W.paired_boot(z.ens_speed, z.ens)[:2],
              "vs standings", W.paired_boot(z.ens_speed, z.standings)[:2])
        d = X[X.season.isin(years)].dropna(subset=["prior_speed_adv", "y_pace"])
        pace = d.groupby(["season", "round"]).apply(
            lambda g: spearmanr(-g.prior_speed_adv, g.y_pace)[0] if len(g) >= 6 else np.nan,
            include_groups=False)
        print(split, "speed forecast vs actual pace rho", round(pace.mean(), 4), "races", pace.notna().sum())
    q = P[P.model.isin(["logit", "logit_speed"])].merge(
        X[["season", "round", "driver", "podium"]], on=["season", "round", "driver"], validate="many_to_one")
    q["brier"] = (-q.score - q.podium) ** 2
    for split, years in (("inner", range(2020, 2023)), ("outer", range(2023, 2026))):
        race = q[q.season.isin(years)].groupby(["season", "round", "model"]).brier.mean().unstack()
        print(split, "logit Brier", round(race.logit.mean(), 6),
              "speed Brier", round(race.logit_speed.mean(), 6),
              "paired difference", W.paired_boot(race.logit_speed, race.logit)[:2])
    picks = P[P.model.isin(["ens", "ens_speed"])].sort_values(["model", "season", "round", "score", "driver"])
    picks = picks.groupby(["model", "season", "round"]).head(3).groupby(["model", "season", "round"]).driver.apply(frozenset)
    print("changed top-three sets", {split: sum(picks["ens", y, r] != picks["ens_speed", y, r]
                                            for y, r in R[R.split == split][["season", "round"]].drop_duplicates().itertuples(index=False, name=None))
                                      for split in ("inner", "outer")})


if __name__ == "__main__":
    main()
