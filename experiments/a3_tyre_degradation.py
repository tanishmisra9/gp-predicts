"""A3 pilot: repeatability of teammate-relative tyre degradation from past races."""
import warnings

import fastf1
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.common import table
from f1.data import DATA, load_results

PART = DATA / "parts/teammate_degradation_2018_2025.csv"


def race_styles(laps, entrants):
    l = laps[(laps.LapNumber > 1) & laps.LapTime.notna() & laps.PitInTime.isna()
             & laps.PitOutTime.isna() & laps.TrackStatus.astype(str).eq("1")
             & laps.IsAccurate.fillna(False) & laps.TyreLife.notna() & laps.Compound.notna()].copy()
    l["t"] = l.LapTime.dt.total_seconds()
    l = l[l.t < 1.07 * l.groupby("Driver").t.transform("median")]
    l["residual"] = l.t - l.groupby("LapNumber").t.transform("median")
    stints = []
    for (code, stint, compound), g in l.groupby(["Driver", "Stint", "Compound"]):
        if len(g) < 8 or g.TyreLife.nunique() < 6:
            continue
        slope = np.polyfit(g.TyreLife, g.residual, 1)[0]
        if np.isfinite(slope) and abs(slope) <= 1:
            stints.append((code, compound, slope, len(g)))
    if not stints:
        return pd.DataFrame(columns=["code", "style", "common_compounds"])
    s = pd.DataFrame(stints, columns=["code", "compound", "slope", "n"])
    s = s.groupby(["code", "compound"]).apply(
        lambda g: pd.Series({"slope": np.average(g.slope, weights=g.n), "n": g.n.sum()}), include_groups=False).reset_index()
    s = s.merge(entrants[["code", "team"]], on="code", validate="many_to_one")
    rows = []
    for team, g in s.groupby("team"):
        codes = g.code.unique()
        if len(codes) != 2:
            continue
        a = g[g.code == codes[0]].set_index("compound")
        b = g[g.code == codes[1]].set_index("compound")
        common = a.index.intersection(b.index)
        if len(common) == 0:
            continue
        w = np.minimum(a.loc[common, "n"], b.loc[common, "n"])
        delta = np.average(a.loc[common, "slope"] - b.loc[common, "slope"], weights=w)
        rows.extend([(codes[0], delta, len(common)), (codes[1], -delta, len(common))])
    return pd.DataFrame(rows, columns=["code", "style", "common_compounds"])


def extract(results):
    fastf1.Cache.enable_cache(str(DATA / "fastf1_cache"))
    fastf1.Cache.offline_mode(True)
    rows = pd.read_csv(PART).to_dict("records") if PART.exists() else []
    done = {(x["season"], x["round"]) for x in rows}
    for (year, round_), entrants in results.groupby(["season", "round"], sort=True):
        if (year, round_) in done:
            continue
        try:
            session = fastf1.get_session(int(year), int(round_), "R")
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            m = race_styles(session.laps, entrants)
            if len(m):
                rows.extend(m.assign(season=year, round=round_).to_dict("records"))
                pd.DataFrame(rows).to_csv(PART, index=False)
            done.add((year, round_))
            print(year, round_, "styles", len(m), flush=True)
        except Exception as exc:
            print(year, round_, "unavailable", type(exc).__name__, str(exc)[:100], flush=True)
    return pd.DataFrame(rows)


def prior_style(X, styles):
    """Previous observed driver style, with no target-race style input."""
    out = X.merge(styles[["season", "round", "code", "style"]],
                  on=["season", "round", "code"], how="left", validate="one_to_one")
    out = out.sort_values(["code", "k"])
    out["prior_style"] = out.groupby("code").style.transform(lambda x: x.ffill().shift(1))
    return out.sort_index()


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    results = load_results(); results = results[results.season <= 2025]
    styles = extract(results)
    X = table(); X = X[X.season <= 2025].copy()
    X = prior_style(X, styles)
    print("style coverage", styles.groupby("season").agg(races=("round", "nunique"), rows=("code", "size")).to_string())
    for name, years in (("inner", [2020, 2021, 2022]), ("outer", [2023, 2024, 2025])):
        d = X[X.season.isin(years)].dropna(subset=["style", "prior_style"])
        print(name, "driver-race pairs", len(d), "prior/current style corr", round(d.prior_style.corr(d["style"]), 4))
        pairs = []
        for (_, _, _), g in d.groupby(["season", "round", "team"]):
            if len(g) != 2 or g.prior_style.nunique() != 2 or g.order.nunique() != 2:
                continue
            g = g.sort_values("prior_style")
            baseline = -g.d_season_pts.fillna(0) * 100 + g.d10_order.fillna(11) * 1e-3
            actual = g.order.iloc[0] < g.order.iloc[1]
            pairs.append((int(actual), int((baseline.iloc[0] < baseline.iloc[1]) == actual)))
        null = np.random.default_rng(0).integers(0, 2, size=(1000, len(pairs))).mean(axis=1) if pairs else np.array([])
        print(name, "teammate pairs", len(pairs), "lower prior degradation beats teammate",
              round(np.mean([p[0] for p in pairs]), 4) if pairs else None,
              "standings on same pairs", round(np.mean([p[1] for p in pairs]), 4) if pairs else None,
              "within-pair label-shuffle null", round(null.mean(), 4) if pairs else None)
        rho = d.groupby(["season", "round"]).apply(
            lambda g: spearmanr(g.prior_style, g["style"])[0] if len(g) >= 6 else np.nan, include_groups=False).mean()
        print(name, "per-race style Spearman", round(rho, 4))


if __name__ == "__main__":
    main()
