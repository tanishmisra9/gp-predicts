"""E174: retrospective winner error attribution and inadmissible hindsight selectors."""
import numpy as np
import pandas as pd

import f1.walkforward as W


def main():
    df = pd.read_csv("data/results.csv").loc[lambda d: d.season.between(2020, 2025)]
    pace = pd.read_csv("data/racepace_ff1.csv").loc[lambda d: d.season.between(2020, 2025)]
    p = pd.read_csv("results/e169_predictions.csv", float_precision="round_trip").loc[lambda d: d.model.eq("ens")]
    d = df.merge(p, on=["season", "round", "driver"], validate="one_to_one")
    d = d.merge(pace[["season", "round", "code", "pace_gap"]], on=["season", "round", "code"], how="left", validate="one_to_one")
    assert len(d) == len(df)
    rows = []; errors = []
    for (season, round_), g in d.groupby(["season", "round"], sort=True):
        g = g.sort_values(["score", "driver"]).copy()
        g["base_rank"] = np.arange(len(g), dtype=float)
        best = g.iloc[0]; actual = g.loc[g.order.eq(1)].iloc[0]
        dnf = ~(g.status.isin(["Finished", "Lapped"]) | g.status.str.startswith("+"))
        winning_team = g.team.eq(actual.team)
        chosen_team = g.team.eq(best.team)
        better_mate = g.loc[chosen_team].sort_values("order").index[0]
        mate_score = g.base_rank.copy(); mate_score.loc[better_mate] = -1.
        scores = dict(
            ens=g.score,
            ORACLE_winning_team=g.base_rank + (~winning_team) * len(g),
            ORACLE_teammate=mate_score,
            ORACLE_retirers=g.base_rank + dnf * len(g),
            ORACLE_pace=g.pace_gap.fillna(99),
            ORACLE_pace_retirers=g.pace_gap.fillna(99) + dnf * 1000,
        )
        for name, score in scores.items():
            metrics = W.metrics(g, score, tie=None)
            chosen = g.assign(oracle_score=score).sort_values(["oracle_score", "driver"]).iloc[0]
            assert metrics["winner"] == float(chosen.driver == actual.driver)
            rows.append(dict(season=season, round=round_, model=name, **metrics))
        errors.append(dict(season=season, round=round_,
                           hit=best.driver == actual.driver,
                           wrong_team=best.team != actual.team,
                           wrong_mate=(best.team == actual.team and best.driver != actual.driver),
                           pace_rows=int(g.pace_gap.notna().sum()), driver_rows=len(g),
                           winner_has_pace=pd.notna(actual.pace_gap)))
    R = pd.DataFrame(rows); E = pd.DataFrame(errors)
    assert (E[["hit", "wrong_team", "wrong_mate"]].sum(axis=1) == 1).all()
    baseline = pd.read_csv("results/e169_per_race.csv").loc[lambda d: d.model.eq("ens")].set_index(["season", "round"])
    current = R[R.model.eq("ens")].set_index(["season", "round"])
    np.testing.assert_allclose(current[["winner", "podium"]], baseline[["winner", "podium"]])
    for split, years in (("inner", [2020, 2021, 2022]), ("outer", [2023, 2024, 2025])):
        r = R[R.season.isin(years)]; e = E[E.season.isin(years)]
        print(split, r.groupby("model")[["winner", "podium"]].agg(["sum", "mean"]).round(4).to_string())
        print("error counts", e[["hit", "wrong_team", "wrong_mate"]].sum().to_dict())
        print("pace coverage", int(e.pace_rows.sum()), "/", int(e.driver_rows.sum()),
              "actual winners with pace", int(e.winner_has_pace.sum()), "/", len(e))
        a = r[r.model.eq("ens")].set_index(["season", "round"]).winner
        for name in sorted(set(r.model) - {"ens"}):
            b = r[r.model.eq(name)].set_index(["season", "round"]).winner
            print(name, "minus ens", W.paired_boot(b, a))
    R.to_csv("results/e174_hindsight_metrics.csv", index=False)
    E.to_csv("results/e174_error_anatomy.csv", index=False)


if __name__ == "__main__":
    main()
