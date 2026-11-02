"""B3: team stop tendency times past circuit pit-lane cost."""

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.common import table
from f1.data import DATA, load_results
from f1.features import pre_weekend_cols


def source_table(results, pits):
    r = results[results.season <= 2025].copy()
    p = pits[pits.season <= 2025].drop_duplicates(["season", "round", "code"])
    z = r.merge(p[["season", "round", "code", "n_pit", "pit_time"]],
                on=["season", "round", "code"], how="left", validate="one_to_one")
    z["race_laps"] = z.groupby(["season", "round"]).laps.transform("max")
    z = z[z.laps >= 0.9 * z.race_laps].copy()  # avoid truncated DNF strategies
    z["stop_resid"] = z.n_pit - z.groupby(["season", "round"]).n_pit.transform("median")
    team = z.groupby(["season", "round", "team", "circuit"]).agg(
        stop_resid=("stop_resid", "mean"), n=("stop_resid", "count")).reset_index()
    cost = z.groupby(["season", "round", "circuit"]).pit_time.median().rename("pit_cost").reset_index()
    return team, cost


def prior_exposure(X, team, cost):
    """All source rows for a target race must have lower event k."""
    events = X[["season", "round", "k", "circuit"]].drop_duplicates()
    t = team.merge(events[["season", "round", "k"]], on=["season", "round"], validate="many_to_one")
    c = cost.merge(events[["season", "round", "k"]], on=["season", "round"], validate="many_to_one")
    out = []
    for k, cur in X.groupby("k", sort=True):
        past_t = t[t.k < k].copy()
        past_t["w"] = past_t.n * 0.5 ** ((k - past_t.k) / 10)
        team_stop = (past_t.stop_resid * past_t.w).groupby(past_t.team).sum() / past_t.w.groupby(past_t.team).sum()
        past_c = c[(c.k < k) & (c.circuit == cur.circuit.iloc[0])]
        pit_cost = past_c.pit_cost.tail(3).median() if len(past_c) else c[c.k < k].pit_cost.median()
        f = cur[["season", "round", "code", "team"]].copy()
        f["prior_team_stops"] = f.team.map(team_stop)
        f["prior_pit_cost"] = pit_cost
        f["strategy_exposure"] = f.prior_team_stops * f.prior_pit_cost
        out.append(f.drop(columns="team"))
    return pd.concat(out, ignore_index=True)


def main():
    results = load_results()
    pits = pd.concat([pd.read_csv(f) for f in sorted((DATA / "parts").glob("pits_*.csv"))], ignore_index=True)
    team, cost = source_table(results, pits)
    X = table()
    X = X[X.season <= 2025].copy()
    circuit = results[["season", "round", "circuit"]].drop_duplicates()
    X = X.merge(circuit, on=["season", "round"], validate="many_to_one")
    f = prior_exposure(X, team, cost)
    X = X.merge(f, on=["season", "round", "code"], validate="one_to_one")
    diagnostics = X[["season", "round", "team", "circuit", "prior_team_stops", "prior_pit_cost"]].drop_duplicates()
    diagnostics = diagnostics.merge(team[["season", "round", "team", "stop_resid"]], on=["season", "round", "team"], how="left")
    diagnostics = diagnostics.merge(cost[["season", "round", "pit_cost"]], on=["season", "round"], how="left")
    for split, years in (("inner", range(2020, 2023)), ("outer", range(2023, 2026))):
        z = diagnostics[diagnostics.season.isin(years)]
        print(split, "team stop tendency forecast/actual correlation", round(z.prior_team_stops.corr(z.stop_resid), 4),
              "coverage", round(z.prior_team_stops.notna().mean(), 4),
              "pit cost forecast/actual correlation", round(z.prior_pit_cost.corr(z.pit_cost), 4))
    base = pre_weekend_cols(X)
    W.assert_pre_weekend(base + ["strategy_exposure"])
    W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
    W.MODELS["logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, base, C=0.03, hl=100)
    W.MODELS["logit_strategy"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, base + ["strategy_exposure"], C=0.03, hl=100)
    def shuffled(tr, te, F, **kw):
        t = tr.copy()
        t["podium"] = np.random.default_rng(0).permutation(t.podium.values)
        return W.m_logit_pod(t, te, base, C=0.03, hl=100)
    W.MODELS["null_shuffled_labels"] = shuffled
    R, _ = W.walk_forward(X, range(2020, 2026),
                          ["standings", "logit", "logit_strategy", "null_shuffled_labels"],
                          ens={"ens": ["standings", "logit"],
                               "ens_strategy": ["standings", "logit_strategy"]}, tie=None)
    R["split"] = np.where(R.season <= 2022, "inner", "outer")
    print("strict overlap", R.groupby(["model", "split"]).podium.mean().unstack().round(4).to_string())
    for split in ("inner", "outer"):
        z = R[R.split == split].pivot(index=["season", "round"], columns="model", values="podium")
        print(split, "strategy ensemble vs standings", W.paired_boot(z.ens_strategy, z.standings)[:2],
              "vs ens", W.paired_boot(z.ens_strategy, z.ens)[:2])


if __name__ == "__main__":
    main()
