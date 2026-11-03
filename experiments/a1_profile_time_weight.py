"""E131: prior-year X/Y profile projected to lap time using fixed physics weights."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import f1.walkforward as W
from experiments.a1_multistint_midfield import DRIVERS, FRONT_PART, PART
from experiments.common import table
from f1.data import load_results


def prior_time_advantage(source):
    """For equal-distance bins, delta lap time is proportional to -delta speed / speed²."""
    field = source.groupby(["circuit", "season", "half", "bin"]).speed.median().rename("field")
    d = source.join(field, on=["circuit", "season", "half", "bin"]).copy()
    d["weighted_adv"] = (d.speed - d.field) / d.field.pow(2)
    p = d.groupby(["circuit", "season", "driver"], as_index=False).weighted_adv.mean()
    p["season"] += 1
    return p.rename(columns={"weighted_adv": "prior_xy_time_adv"})


def main():
    source = pd.concat([pd.read_csv(FRONT_PART), pd.read_csv(PART)], ignore_index=True)
    prior = prior_time_advantage(source)
    prior["code"] = prior.driver.map(DRIVERS)
    results = load_results()
    circuits = results[["season", "round", "circuit"]].drop_duplicates()
    names = {"bahrain": "Bahrain", "red_bull_ring": "Austria", "monza": "Monza"}
    X = table().merge(circuits, on=["season", "round"], validate="many_to_one")
    X = X[(X.season == 2024) & X.circuit.isin(names)].copy()
    X["circuit_name"] = X.circuit.map(names)
    d = X.merge(prior[["circuit", "season", "code", "prior_xy_time_adv"]],
                left_on=["circuit_name", "season", "code"],
                right_on=["circuit", "season", "code"], validate="one_to_one")
    d = d.dropna(subset=["y_pace", "prior_xy_time_adv"]).copy()
    d["standings_score"] = -d.d_season_pts.fillna(0) * 100 + d.d10_order.fillna(11) * 1e-3
    d["profile_score"] = -d.prior_xy_time_adv
    d["blend_score"] = (d.groupby("circuit_name").standings_score.rank() +
                        d.groupby("circuit_name").profile_score.rank()) / 2
    rows = []
    for circuit, g in d.groupby("circuit_name"):
        rows.append({"circuit": circuit, "drivers": len(g), **{
            name: spearmanr(g[name + "_score"], g.y_pace).statistic
            for name in ("standings", "profile", "blend")}})
    r = pd.DataFrame(rows).set_index("circuit")
    print("2024 pace-order Spearman on common continuing drivers", r.round(4).to_string())
    print("mean", r.drop(columns="drivers").mean().round(4).to_dict())
    for candidate in ("profile", "blend"):
        print(candidate, "minus standings paired race bootstrap", W.paired_boot(r[candidate], r.standings)[:2])
    rng = np.random.default_rng(131)
    observed = r.profile.mean()
    null = np.array([np.mean([spearmanr(g.profile_score, rng.permutation(g.y_pace)).statistic
                              for _, g in d.groupby("circuit_name")]) for _ in range(5000)])
    print("within-race pace-label shuffle mean", round(null.mean(), 4),
          "one-sided p", round((1 + (null >= observed).sum()) / (len(null) + 1), 4))
    print("strict podium reference: standings 0.6066 inner / 0.5905 outer; frozen ens 0.6284 / 0.5857; shuffled-label logit null 0.1749 / 0.1857")


if __name__ == "__main__":
    main()
