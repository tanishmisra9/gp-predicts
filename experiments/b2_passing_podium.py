"""E153: one past-only passing-history feature in the frozen podium logit."""

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.b2_clean_passing import PART
from f1.data import DATA, load_results
from f1.features import build_features, pre_weekend_cols


def prior_driver(results, source):
    events = results[["season", "round", "date"]].drop_duplicates().sort_values(["date", "season", "round"])
    events["k"] = np.arange(len(events))
    history = source.merge(events[["season", "round", "k"]], on=["season", "round"], validate="many_to_one")
    entrants = results[["season", "round", "code"]].drop_duplicates().merge(
        events[["season", "round", "k"]], on=["season", "round"], validate="many_to_one")
    out = []
    for k, target in entrants.groupby("k", sort=True):
        past = history[history.k < k].copy()
        past["w"] = 0.5 ** ((k - past.k) / 50)
        global_rate = (past.w * past.converted).sum() / past.w.sum() if len(past) else 0
        wins = (past.w * past.converted).groupby(past.trailing).sum()
        exposures = past.w.groupby(past.trailing).sum()
        rate = (wins + 200 * global_rate) / (exposures + 200)
        out.append(target[["season", "round", "code"]].assign(
            prior_pass=target.code.map(rate).fillna(global_rate).values))
    return pd.concat(out, ignore_index=True)


def main():
    results = load_results().query("season <= 2025")
    pace = pd.read_csv(DATA / "racepace_ff1.csv").query("season <= 2025")
    X = build_features(results, pace)
    base = pre_weekend_cols(X)
    source = pd.read_csv(PART)
    feature = prior_driver(results, source)
    X = X.merge(feature, on=["season", "round", "code"], validate="one_to_one")
    W.assert_pre_weekend(base + ["prior_pass"])
    W.MODELS["standings"] = lambda tr, te, f, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
    W.MODELS["logit"] = lambda tr, te, f, **kw: W.m_logit_pod(tr, te, base, C=0.03, hl=100)
    W.MODELS["logit_pass"] = lambda tr, te, f, **kw: W.m_logit_pod(tr, te, base + ["prior_pass"], C=0.03, hl=100)

    def shuffled(tr, te, f, **kw):
        t = tr.copy()
        t["podium"] = np.random.default_rng(0).permutation(t.podium.values)
        return W.m_logit_pod(t, te, base, C=0.03, hl=100)

    W.MODELS["null_shuffled_labels"] = shuffled
    R, P = W.walk_forward(X, range(2020, 2026),
                          ["standings", "logit", "logit_pass", "null_shuffled_labels"],
                          ens={"ens": ["standings", "logit"],
                               "ens_pass": ["standings", "logit_pass"]}, tie=None)
    R["split"] = np.where(R.season <= 2022, "inner", "outer")
    print("strict overlap", R.groupby(["model", "split"]).podium.mean().unstack().round(6).to_string())
    print("strict by season", R.groupby(["model", "season"]).podium.mean().unstack().round(6).to_string())
    picks = P[P.model.isin(["ens", "ens_pass"])].sort_values(["model", "season", "round", "score", "driver"])
    picks = picks.groupby(["model", "season", "round"]).head(3)
    picks = picks.groupby(["model", "season", "round"]).driver.apply(frozenset)
    for split in ("inner", "outer"):
        z = R[R.split == split].pivot(index=["season", "round"], columns="model", values="podium")
        changed = sum(picks["ens", year, round_] != picks["ens_pass", year, round_] for year, round_ in z.index)
        print(split, "races", len(z), "changed top-three sets", changed)
        for candidate, reference in (("ens_pass", "standings"), ("ens_pass", "ens"),
                                     ("logit_pass", "logit")):
            print(split, candidate, "minus", reference, W.paired_boot(z[candidate], z[reference])[:2])
    probs = P[P.model.isin(["logit", "logit_pass"])].merge(
        X[["season", "round", "driver", "podium"]], on=["season", "round", "driver"], validate="many_to_one")
    p = (-probs.score).clip(1e-6, 1 - 1e-6)
    probs["brier"] = (p - probs.podium) ** 2
    probs["logloss"] = -(probs.podium * np.log(p) + (1 - probs.podium) * np.log1p(-p))
    probs["split"] = np.where(probs.season <= 2022, "inner", "outer")
    print("probability metrics", probs.groupby(["model", "split"])[["brier", "logloss"]].mean().round(6).to_string())
    by_race = probs.groupby(["split", "season", "round", "model"])[["brier", "logloss"]].mean().reset_index()
    for split in ("inner", "outer"):
        z = by_race[by_race.split == split].pivot(index=["season", "round"], columns="model", values=["brier", "logloss"])
        for metric in ("brier", "logloss"):
            print(split, metric, "augmented minus frozen", W.paired_boot(z[metric].logit_pass, z[metric].logit)[:2])


if __name__ == "__main__":
    main()
