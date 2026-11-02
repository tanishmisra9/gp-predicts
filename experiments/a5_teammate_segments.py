"""A5: same-lap, same-compound, similar-age teammate race-pace contrasts."""

import warnings

import fastf1
import numpy as np
import pandas as pd

import f1.ff1  # initialize the project's existing FastF1 cache
import f1.walkforward as W
from experiments.common import table
from f1.data import DATA, load_results
from f1.features import pre_weekend_cols


PART = DATA / "parts/teammate_segments_2018_2025.csv"


def race_pairs(laps, entrants):
    line = laps[laps.Time.notna() & laps.Position.notna()].sort_values(["LapNumber", "Position"]).copy()
    line["ahead_time"] = line.groupby("LapNumber").Time.shift(1)
    line["gap_ahead"] = (line.Time - line.ahead_time).dt.total_seconds()
    l = line[(line.LapNumber > 1) & line.LapTime.notna() & line.PitInTime.isna()
             & line.PitOutTime.isna() & line.TrackStatus.astype(str).eq("1")
             & line.IsAccurate.fillna(False) & line.TyreLife.notna() & line.Compound.notna()
             & ((line.Position == 1) | (line.gap_ahead > 1.5))].copy()
    l["t"] = l.LapTime.dt.total_seconds()
    l = l[l.t < 1.07 * l.groupby("Driver").t.transform("median")]
    l = l.merge(entrants[["code", "team"]].rename(columns={"code": "Driver"}),
                on="Driver", validate="many_to_one")
    rows = []
    for team, g in l.groupby("team"):
        codes = sorted(g.Driver.unique())
        if len(codes) != 2:
            continue
        a = g[g.Driver == codes[0]][["LapNumber", "Compound", "TyreLife", "t"]]
        b = g[g.Driver == codes[1]][["LapNumber", "Compound", "TyreLife", "t"]]
        m = a.merge(b, on=["LapNumber", "Compound"], suffixes=("_a", "_b"), validate="one_to_one")
        m = m[(m.TyreLife_a - m.TyreLife_b).abs() <= 3]
        if len(m) < 5:
            continue
        gap = 100 * (m.t_a - m.t_b).median() / l.t.median()
        rows.extend([(team, codes[0], codes[1], gap, len(m)),
                     (team, codes[1], codes[0], -gap, len(m))])
    return pd.DataFrame(rows, columns=["team", "code", "mate", "gap", "n"])


def extract(results):
    fastf1.Cache.offline_mode(True)
    rows = pd.read_csv(PART).to_dict("records") if PART.exists() else []
    done = {(x["season"], x["round"]) for x in rows}
    for (year, round_), entrants in results.groupby(["season", "round"], sort=True):
        if (year, round_) in done:
            continue
        try:
            session = fastf1.get_session(int(year), int(round_), "R")
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            m = race_pairs(session.laps, entrants)
            if len(m):
                rows.extend(m.assign(season=year, round=round_).to_dict("records"))
                pd.DataFrame(rows).to_csv(PART, index=False)
            done.add((year, round_))
            print(year, round_, "paired teams", len(m) // 2, flush=True)
        except Exception as exc:
            print(year, round_, "unavailable", type(exc).__name__, str(exc)[:100], flush=True)
    return pd.DataFrame(rows)


def prior_pair(rows, events):
    """Past-only ten-race-half-life matched pace for the same teammate pair."""
    keys = rows[["season", "round", "k"]].drop_duplicates()
    source = events.merge(keys, on=["season", "round"], validate="many_to_one")
    out = []
    for k, cur in rows.groupby("k", sort=True):
        past = source[source.k < k].copy()
        past["w"] = past.n * 0.5 ** ((k - past.k) / 10)
        key = ["team", "code", "mate"]
        g = (past.gap * past.w).groupby([past[c] for c in key]).sum() / past.w.groupby([past[c] for c in key]).sum()
        f = cur[["season", "round", "team", "code", "mate"]].copy()
        f["prior_gap"] = pd.MultiIndex.from_frame(f[key]).map(g)
        out.append(f)
    return pd.concat(out, ignore_index=True)


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    results = load_results()
    results = results[results.season <= 2025]
    events = extract(results)
    X = table()
    X = X[X.season <= 2025]
    pair_rows = X[["season", "round", "k", "team", "code", "driver", "order", "d_season_pts", "d10_order"]].copy()
    mates = pair_rows[["season", "round", "team", "code"]].merge(
        pair_rows[["season", "round", "team", "code"]], on=["season", "round", "team"], suffixes=("", "_mate"))
    mates = mates[mates.code != mates.code_mate].rename(columns={"code_mate": "mate"})
    pair_rows = pair_rows.merge(mates[["season", "round", "code", "mate"]], on=["season", "round", "code"], validate="one_to_one")
    f = prior_pair(pair_rows, events)
    d = pair_rows.merge(f, on=["season", "round", "team", "code", "mate"], validate="one_to_one")
    d = d.merge(events[["season", "round", "code", "gap"]].rename(columns={"gap": "current_gap"}),
                on=["season", "round", "code"], how="left", validate="one_to_one")
    base = pre_weekend_cols(X)
    W.assert_pre_weekend(base + ["prior_gap"])
    W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
    W.MODELS["logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, base, C=0.03, hl=100)
    def shuffled(tr, te, F, **kw):
        t = tr.copy()
        t["podium"] = np.random.default_rng(0).permutation(t.podium.values)
        return W.m_logit_pod(t, te, base, C=0.03, hl=100)
    W.MODELS["null_shuffled_labels"] = shuffled
    R, P = W.walk_forward(X, range(2020, 2026),
                          ["standings", "logit", "null_shuffled_labels"],
                          ens={"ens": ["standings", "logit"]}, tie=None)
    frozen = P[P.model == "ens"]
    revised = []
    pair_comparisons = []
    changed_top3 = []
    for (season, round_), g in frozen.groupby(["season", "round"]):
        target = X[(X.season == season) & (X["round"] == round_)].copy()
        scores = g.set_index("driver").score.reindex(target.driver).copy()
        scores.index = target.driver.values
        original_scores = scores.copy()
        pairs_now = d[(d.season == season) & (d["round"] == round_)]
        for team, team_pair in pairs_now.groupby("team"):
            if len(team_pair) != 2 or team_pair.prior_gap.isna().any():
                continue
            a, b = team_pair.iloc[0], team_pair.iloc[1]
            if not np.isclose(a.prior_gap, -b.prior_gap):
                raise ValueError("asymmetric teammate forecast")
            old_a, old_b = scores.loc[a.driver], scores.loc[b.driver]
            faster_a = a.prior_gap < 0
            lo, hi = sorted((old_a, old_b))
            if lo == hi:
                lo -= 1e-8; hi += 1e-8
            scores.loc[a.driver], scores.loc[b.driver] = (lo, hi) if faster_a else (hi, lo)
            actual_a = a.order < b.order
            old_correct = float((old_a < old_b) == actual_a) if old_a != old_b else np.nan
            pair_comparisons.append(dict(season=season, round=round_, team=team,
                                         correct=int(faster_a == actual_a), old_correct=old_correct,
                                         podium=int((team_pair.order <= 3).any())))
        revised.append(dict(season=season, round=round_, model="ens_teammate_swap",
                            **W.metrics(target, target.driver.map(scores).values, tie=None)))
        top3 = lambda s: frozenset(s.rename_axis("driver").reset_index(name="score").sort_values(["score", "driver"]).head(3).driver)
        changed_top3.append((season, top3(original_scores) != top3(scores)))
    R = pd.concat([R, pd.DataFrame(revised)], ignore_index=True)
    R["split"] = np.where(R.season <= 2022, "inner", "outer")
    print("strict overlap", R.groupby(["model", "split"]).podium.mean().unstack().round(4).to_string())
    for split in ("inner", "outer"):
        z = R[R.split == split].pivot(index=["season", "round"], columns="model", values="podium")
        print(split, "swap vs standings", W.paired_boot(z.ens_teammate_swap, z.standings)[:2],
              "vs ens", W.paired_boot(z.ens_teammate_swap, z.ens)[:2],
              "top3 changes", sum(changed for year, changed in changed_top3
                                  if (year <= 2022) == (split == "inner")))
    comp = pd.DataFrame(pair_comparisons)
    for split, years in (("inner", range(2020, 2023)), ("outer", range(2023, 2026))):
        c = comp[comp.season.isin(years)]
        for label, subset in (("all", c), ("podium", c[c.podium == 1])):
            print(split, label, "covered pairs", len(subset), "pace/ens call",
                  round(subset.correct.mean(), 4), round(subset.old_correct.mean(), 4))
            by_race = subset.groupby(["season", "round"])[["correct", "old_correct"]].mean()
            print(split, label, "pair accuracy diff vs ens, race bootstrap",
                  W.paired_boot(by_race.correct, by_race.old_correct)[:2])
    print("coverage", events.groupby("season").agg(races=("round", "nunique"), driver_rows=("code", "size")).to_string())
    for split, years in (("inner", range(2020, 2023)), ("outer", range(2023, 2026))):
        z = d[d.season.isin(years)].dropna(subset=["prior_gap", "current_gap"])
        print(split, "driver rows", len(z), "prior/current correlation", round(z.prior_gap.corr(z.current_gap), 4))
        pairs = []
        for _, g in d[d.season.isin(years)].dropna(subset=["prior_gap"]).groupby(["season", "round", "team"]):
            if len(g) != 2:
                continue
            g = g.sort_values("prior_gap")
            actual = g.order.iloc[0] < g.order.iloc[1]
            standings = -g.d_season_pts.fillna(0) * 100 + g.d10_order.fillna(11) * 1e-3
            pairs.append((int(actual), int((standings.iloc[0] < standings.iloc[1]) == actual),
                          int((g.order <= 3).any())))
        p = np.asarray(pairs)
        if len(p):
            print(split, "pairs", len(p), "prior-pair pace picks winner", round(p[:, 0].mean(), 4),
                  "standings", round(p[:, 1].mean(), 4),
                  "podium-involved pairs", int(p[:, 2].sum()),
                  "pace/standings there", round(p[p[:, 2] == 1, 0].mean(), 4), round(p[p[:, 2] == 1, 1].mean(), 4))


if __name__ == "__main__":
    main()
