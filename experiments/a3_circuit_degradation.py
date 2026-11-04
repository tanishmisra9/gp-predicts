"""E150: does a team's historical tyre degradation repeat at the same circuit?"""

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.a3_team_degradation import PART, prior_team
from f1.data import load_results


def circuit_forecast(results, source):
    events = results[["season", "round", "date", "circuit"]].drop_duplicates().sort_values(["date", "season", "round"])
    events["k"] = np.arange(len(events))
    history = source.merge(events[["season", "round", "circuit", "k"]],
                           on=["season", "round"], validate="many_to_one")
    target = prior_team(results, source).merge(events[["season", "round", "circuit", "k"]],
                                                on=["season", "round"], validate="many_to_one")
    rows = []
    for row in target.itertuples(index=False):
        past = history[(history.k < row.k) & (history.team == row.team) & (history.circuit == row.circuit)]
        weights = 0.5 ** (row.season - past.season - 1)
        value = ((past.relative_deg * weights).sum() + 3 * row.prior_deg) / (weights.sum() + 3)
        rows.append((value, len(past)))
    target[["circuit_deg", "n_same_circuit"]] = rows
    return target


def main():
    results = load_results().query("season <= 2025")
    source = pd.read_csv(PART)
    forecast = circuit_forecast(results, source)
    truth = source.merge(forecast, on=["season", "round", "team"], validate="one_to_one")
    truth = truth[truth.season.between(2020, 2025)].copy()
    for name, p in (("circuit", truth.circuit_deg), ("global", truth.prior_deg), ("zero", 0)):
        truth[name + "_mse"] = (p - truth.relative_deg) ** 2
    print("coverage", truth.groupby("season").agg(races=("round", "nunique"), teams=("team", "size"),
                                                  same_circuit=("n_same_circuit", lambda x: int((x > 0).sum()))).to_string())
    print("per-season MSE", truth.groupby("season")[["circuit_mse", "global_mse", "zero_mse"]].mean().round(7).to_string())
    for label, years in (("inner", range(2020, 2023)), ("outer", range(2023, 2026))):
        g = truth[truth.season.isin(years)]
        race = g.groupby(["season", "round"])[["circuit_mse", "global_mse", "zero_mse"]].mean()
        print(label, "team-races", len(g), "races", len(race),
              "same-circuit source", int((g.n_same_circuit > 0).sum()))
        print(label, "MSE", g[["circuit_mse", "global_mse", "zero_mse"]].mean().round(7).to_dict())
        for ref in ("global", "zero"):
            print(label, "circuit minus", ref, W.paired_boot(race.circuit_mse, race[ref + "_mse"])[:2])
        print(label, "correlations", {name: g[name].corr(g.relative_deg)
                                      for name in ("circuit_deg", "prior_deg")})
        rng = np.random.default_rng(150)
        actual = g.relative_deg.to_numpy()
        predicted = g.circuit_deg.to_numpy()
        groups = list(g.reset_index(drop=True).groupby(["season", "round"]).indices.values())
        null = np.empty(5000)
        for i in range(len(null)):
            shuffled = actual.copy()
            for ix in groups:
                shuffled[ix] = rng.permutation(shuffled[ix])
            null[i] = np.corrcoef(predicted, shuffled)[0, 1]
        observed = g.circuit_deg.corr(g.relative_deg)
        print(label, "within-race shuffled correlation null", null.mean(),
              "one-sided p", (1 + (null >= observed).sum()) / (len(null) + 1))


if __name__ == "__main__":
    main()
