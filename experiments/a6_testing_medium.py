"""A6 pilot: comparable medium-tyre testing laps before rounds 1-3."""
import warnings

import fastf1
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import f1.walkforward as W
from experiments.common import table
from f1.data import DATA
from f1.features import pre_weekend_cols

PART = DATA / "parts/testing_medium_2023_2025.csv"


def medium_day(laps):
    l = laps[(laps.Compound == "MEDIUM") & laps.TyreLife.between(4, 12)
             & laps.LapTime.notna() & laps.PitInTime.isna() & laps.PitOutTime.isna()
             & laps.IsAccurate.fillna(False)].copy()
    l["t"] = l.LapTime.dt.total_seconds()
    g = l.groupby("Driver").t
    p = g.median().where(g.size() >= 6).dropna()
    return ((p / p.min() - 1) * 100).rename("medium_gap") if len(p) else p.rename("medium_gap")


def extract():
    fastf1.Cache.enable_cache(str(DATA / "fastf1_cache"))
    fastf1.Cache.offline_mode(True)
    rows = pd.read_csv(PART).to_dict("records") if PART.exists() else []
    done = {(x["season"], x["day"]) for x in rows}
    for season in (2023, 2024, 2025):
        for day in (1, 2, 3):
            if (season, day) in done:
                continue
            try:
                s = fastf1.get_testing_session(season, 1, day)
                s.load(laps=True, telemetry=False, weather=False, messages=False)
                p = medium_day(s.laps)
                rows.extend(p.reset_index().rename(columns={"Driver": "code"}).assign(season=season, day=day).to_dict("records"))
                pd.DataFrame(rows).to_csv(PART, index=False)
                print(season, day, "drivers", len(p), flush=True)
            except Exception as exc:
                print(season, day, "unavailable", type(exc).__name__, str(exc)[:100], flush=True)
    return pd.DataFrame(rows)


def attach_testing(X, medium, best):
    """Only pre-season testing summaries and current entry-list team are used."""
    m = medium.groupby(["season", "code"]).medium_gap.mean().rename("test_medium")
    b = best.groupby(["season", "code"]).best_gap.mean().rename("test_best")
    out = X.join(m, on=["season", "code"]).join(b, on=["season", "code"])
    for c in ("test_medium", "test_best"):
        out[c] = out.groupby(["season", "round", "team"])[c].transform("mean")
    return out


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    medium = extract()
    best = pd.concat([pd.read_csv(p) for p in (DATA / "parts").glob("testing_*.csv")], ignore_index=True)
    best = best[best.season <= 2025]
    X = table(); X = X[X.season <= 2025].copy()
    X = attach_testing(X, medium, best)
    base = pre_weekend_cols(X)
    W.assert_pre_weekend(base + ["test_medium", "test_best"])
    print("testing rows by year", medium.groupby("season").agg(days=("day", "nunique"), drivers=("code", "nunique")).to_string())
    W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
    W.MODELS["logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, base, C=0.03, hl=100)
    W.MODELS["test_medium"] = lambda tr, te, F, **kw: te.test_medium.fillna(te.test_medium.max() + 1).values
    W.MODELS["test_best"] = lambda tr, te, F, **kw: te.test_best.fillna(te.test_best.max() + 1).values
    def shuffled(tr, te, F, **kw):
        t = tr.copy(); t["podium"] = np.random.default_rng(0).permutation(t.podium.values)
        return W.m_logit_pod(t, te, base, C=0.03, hl=100)
    W.MODELS["null_shuffled_labels"] = shuffled
    R, _ = W.walk_forward(X, [2023, 2024, 2025],
                          ["standings", "logit", "test_medium", "test_best", "null_shuffled_labels"],
                          ens={"ens": ["standings", "logit"],
                               "ens_plus_medium": ["standings", "logit", "test_medium"],
                               "ens_plus_best": ["standings", "logit", "test_best"]}, tie=None)
    R = R[R["round"] <= 3]
    d = R.pivot(index=["season", "round"], columns="model", values="podium")
    print("strict overlap, rounds 1-3 only, 2023-25", d.mean().round(4).to_string())
    for candidate in ("ens_plus_medium", "ens_plus_best"):
        print(candidate, "vs standings", W.paired_boot(d[candidate], d.standings)[:2],
              "vs ens", W.paired_boot(d[candidate], d.ens)[:2])
    print("by season", R.groupby(["model", "season"]).podium.mean().unstack().round(4).to_string())
    for c in ("test_medium", "test_best"):
        z = X[(X.season.between(2023, 2025)) & (X["round"] <= 3)]
        z = z.groupby(["season", "round", "team"]).agg(v=(c, "mean"), pace=("y_pace", "mean")).dropna()
        rho = z.groupby(["season", "round"]).apply(
            lambda g: spearmanr(g.v, g.pace)[0] if len(g) >= 6 else np.nan, include_groups=False).mean()
        print(c, "team race-pace rho", round(rho, 4), "team-race coverage", len(z))


if __name__ == "__main__":
    main()
