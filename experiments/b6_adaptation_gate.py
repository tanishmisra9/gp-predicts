"""E139: does a past-only team-switch adaptation adjustment forecast relative pace?"""

import numpy as np
import pandas as pd

import f1.walkforward as W
from f1.data import DATA, load_results


LINEAGE = {"toro_rosso": "rb", "alphatauri": "rb", "force_india": "aston_martin",
           "racing_point": "aston_martin", "renault": "alpine", "alfa": "sauber"}


def transfer_windows(results, pace):
    r = results[results.season <= 2025].sort_values(["date", "season", "round"]).copy()
    r["lineage"] = r.team.replace(LINEAGE)
    previous = r.groupby("code").lineage.shift()
    r["switch"] = previous.notna() & r.lineage.ne(previous)
    r["spell"] = r.groupby("code").switch.cumsum()
    r["entry"] = r.groupby(["code", "spell"]).cumcount() + 1
    d = r.merge(pace[["season", "round", "code", "pace_gap"]],
                on=["season", "round", "code"], how="left", validate="one_to_one")
    mates = d[["season", "round", "team", "code", "pace_gap"]].rename(
        columns={"code": "mate", "pace_gap": "mate_gap"})
    d = d.merge(mates, on=["season", "round", "team"], how="left")
    d = d[d.code.ne(d.mate)].copy()
    d["relative"] = d.pace_gap - d.mate_gap
    rows = []
    for (code, spell), g in d[d.spell > 0].groupby(["code", "spell"]):
        early = g[g.entry.between(1, 5)].dropna(subset=["relative"])
        late = g[g.entry.between(6, 10)].dropna(subset=["relative"])
        earlier_teams = r[(r.code == code) & (r.date < g.date.min())].lineage
        if (len(early) < 3 or len(late) < 3 or
                len(set(early.mate) | set(late.mate)) != 1 or g.lineage.iloc[0] in set(earlier_teams)):
            continue
        rows.append(dict(code=code, season=int(g.season.iloc[0]), team=g.lineage.iloc[0],
                         early=early.relative.mean(), late=late.relative.mean(),
                         target_date=late.date.min(), end_date=late.date.max(),
                         n_early=len(early), n_late=len(late)))
    return pd.DataFrame(rows).sort_values(["target_date", "code"]).reset_index(drop=True)


def forecast(windows):
    out = windows.copy()
    out["delta"] = out.late - out.early
    out["n_prior"] = [int((out.end_date < date).sum()) for date in out.target_date]
    out["adjustment"] = [out.loc[out.end_date < date, "delta"].sum() / (10 + n)
                         for date, n in zip(out.target_date, out.n_prior)]
    out["adapted"] = out.early + out.adjustment
    return out


def main():
    r = load_results()
    p = pd.read_csv(DATA / "racepace_ff1.csv")
    z = forecast(transfer_windows(r, p))
    prior = r[r.season <= 2025].sort_values(["season", "round"]).copy()
    prior["prior_points"] = prior.groupby(["season", "code"]).points.cumsum() - prior.points
    points_correct = []
    for row in z.itertuples():
        cur = prior[(prior.code == row.code) & (prior.date == row.target_date)].iloc[0]
        mate = prior[(prior.season == cur.season) & (prior["round"] == cur["round"])
                     & (prior.team == cur.team) & (prior.code != cur.code)].iloc[0]
        predicted_faster = (cur.prior_points > mate.prior_points or
                            (cur.prior_points == mate.prior_points and cur.driver < mate.driver))
        points_correct.append(int(predicted_faster == (row.late < 0)))
    z["points_correct"] = points_correct
    z["base_sq"] = (z.early - z.late) ** 2
    z["adapted_sq"] = (z.adapted - z.late) ** 2
    z["base_correct"] = (z.early.lt(0) == z.late.lt(0)).astype(int)
    z["adapted_correct"] = (z.adapted.lt(0) == z.late.lt(0)).astype(int)
    z["split"] = np.select([z.season.eq(2019), z.season.le(2022)], ["2019", "inner"], default="outer")
    for split, g in z.groupby("split"):
        print(split, "transfers", len(g), "mean actual adaptation", round(g.delta.mean(), 4),
              "improved", int(g.delta.lt(0).sum()), "prior counts", (int(g.n_prior.min()), int(g.n_prior.max())))
        print("  MSE base/adapted", g[["base_sq", "adapted_sq"]].mean().round(5).to_dict(),
              "paired adapted-minus-base", W.paired_boot(g.adapted_sq, g.base_sq)[:2])
        print("  faster-teammate accuracy base/adapted/prior-points",
              g[["base_correct", "adapted_correct", "points_correct"]].mean().round(4).to_dict(),
              "adapted-minus-points paired", W.paired_boot(g.adapted_correct, g.points_correct)[:2])
    print(z[["code", "season", "early", "late", "n_prior", "adjustment", "base_sq", "adapted_sq"]]
          .round(4).to_string(index=False))
    print("strict podium reference: standings 0.6066 inner / 0.5905 outer; frozen ens 0.6284 / 0.5857; shuffled-label logit null 0.1749 / 0.1857")


if __name__ == "__main__":
    main()
