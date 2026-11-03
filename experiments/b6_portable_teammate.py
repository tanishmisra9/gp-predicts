"""E116: can matched teammate race pace transfer through driver changes?"""

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.a5_teammate_segments import PART
from experiments.common import table
from f1.features import pre_weekend_cols


def feature_table(X, events):
    """Fit a ridge-anchored driver pace graph from strictly earlier teammate pairs."""
    keys = X[["season", "round", "k"]].drop_duplicates()
    source = events[events.code < events.mate].merge(keys, on=["season", "round"], validate="many_to_one")
    drivers = sorted(set(X.code))
    index = {code: i for i, code in enumerate(drivers)}
    design = np.zeros((len(source), len(drivers)))
    row = np.arange(len(source))
    design[row, source.code.map(index).to_numpy()] = 1
    design[row, source.mate.map(index).to_numpy()] = -1
    out = X.copy()
    out["prior_portable_gap"] = np.nan
    for k, cur in X.groupby("k", sort=True):
        use = source.k.to_numpy() < k
        if not use.any():
            continue
        a = design[use]
        p = source.loc[use]
        w = p.n.to_numpy() * 0.5 ** ((k - p.k.to_numpy()) / 10)
        gram = a.T @ (w[:, None] * a) + 50 * np.eye(len(drivers))
        rating = np.linalg.solve(gram, a.T @ (w * p.gap.to_numpy()))
        seen = set(p.code) | set(p.mate)
        out.loc[cur.index, "prior_portable_gap"] = [rating[index[c]] if c in seen else np.nan for c in cur.code]
    return out


def main():
    X = table(); X = X[X.season <= 2025].copy()
    base = pre_weekend_cols(X)
    source = pd.read_csv(PART)
    X = feature_table(X, source)
    W.assert_pre_weekend(base + ["prior_portable_gap"])
    W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
    W.MODELS["logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, base, C=0.03, hl=100)
    W.MODELS["logit_portable"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, base + ["prior_portable_gap"], C=0.03, hl=100)

    def shuffled(tr, te, F, **kw):
        t = tr.copy()
        t["podium"] = np.random.default_rng(0).permutation(t.podium.values)
        return W.m_logit_pod(t, te, base, C=0.03, hl=100)

    W.MODELS["null_shuffled_labels"] = shuffled
    R, P = W.walk_forward(X, range(2020, 2026),
                          ["standings", "logit", "logit_portable", "null_shuffled_labels"],
                          ens={"ens": ["standings", "logit"],
                               "ens_portable": ["standings", "logit_portable"]}, tie=None)
    R["split"] = np.where(R.season <= 2022, "inner", "outer")
    print("strict overlap", R.groupby(["model", "split"]).podium.mean().unstack().round(4).to_string())
    picks = P[P.model.isin(["ens", "ens_portable"])].sort_values(["model", "season", "round", "score", "driver"])
    picks = picks.groupby(["model", "season", "round"]).head(3)
    picks = picks.groupby(["model", "season", "round"]).driver.apply(frozenset)
    for split in ("inner", "outer"):
        z = R[R.split == split].pivot(index=["season", "round"], columns="model", values="podium")
        changed = sum(picks["ens", y, r] != picks["ens_portable", y, r] for y, r in z.index)
        print(split, "coverage", round(X[X.season.isin(range(2020, 2023) if split == "inner" else range(2023, 2026))].prior_portable_gap.notna().mean(), 4),
              "top3 changes", changed, "portable vs ens", W.paired_boot(z.ens_portable, z.ens)[:2],
              "vs standings", W.paired_boot(z.ens_portable, z.standings)[:2])
    p = P[P.model.isin(["logit", "logit_portable"])].merge(
        X[["season", "round", "driver", "podium"]], on=["season", "round", "driver"], validate="many_to_one")
    p["prob"] = (-p.score).clip(1e-6, 1 - 1e-6)
    p["brier"] = (p.prob - p.podium) ** 2
    p["logloss"] = -(p.podium * np.log(p.prob) + (1 - p.podium) * np.log1p(-p.prob))
    p["split"] = np.where(p.season <= 2022, "inner", "outer")
    print("probabilities", p.groupby(["model", "split"])[["brier", "logloss"]].mean().round(5).to_string())
    annual = p.groupby(["season", "model"])["brier"].mean().unstack()
    print("annual Brier portable minus frozen", (annual.logit_portable - annual.logit).round(5).to_string())
    by_race = p.groupby(["split", "season", "round", "model"])[["brier", "logloss"]].mean().reset_index()
    for split in ("inner", "outer"):
        z = by_race[by_race.split == split].pivot(index=["season", "round"], columns="model", values=["brier", "logloss"])
        for metric in ("brier", "logloss"):
            print(split, metric, "portable minus frozen", W.paired_boot(z[metric].logit_portable, z[metric].logit)[:2])
    r = X.sort_values("k").copy()
    lineage = {"toro_rosso": "rb", "alphatauri": "rb", "force_india": "aston_martin",
               "racing_point": "aston_martin", "renault": "alpine", "alfa": "sauber"}
    r["lineage"] = r.team.replace(lineage)
    r["previous_lineage"] = r.groupby("code").lineage.shift()
    r["transfer"] = r.previous_lineage.notna() & (r.lineage != r.previous_lineage)
    mates = r[["season", "round", "team", "code", "order", "transfer", "prior_portable_gap"]].merge(
        r[["season", "round", "team", "code", "order", "transfer", "prior_portable_gap"]],
        on=["season", "round", "team"], suffixes=("", "_mate"))
    mates = mates[(mates.code < mates.code_mate) & (mates.transfer | mates.transfer_mate)].copy()
    # Transfer diagnostics use only pairs with an observed same-race matched gap.
    observed = source[source.code < source.mate]
    pairs = mates.rename(columns={"code_mate": "mate"}).merge(
        observed[["season", "round", "team", "code", "mate", "gap"]],
        on=["season", "round", "team", "code", "mate"], how="inner")
    pairs = pairs[pairs.season.between(2020, 2025)].dropna(subset=["prior_portable_gap", "prior_portable_gap_mate"])
    pairs["correct"] = ((pairs.prior_portable_gap - pairs.prior_portable_gap_mate < 0) == (pairs.gap < 0)).astype(int)
    print("transfer-pair matched-gap coverage", pairs.groupby(np.where(pairs.season <= 2022, "inner", "outer")).correct.agg(["size", "mean"]).to_string())


if __name__ == "__main__":
    main()
