"""E149: fixed 2025 Kalshi probability diagnostic, no selector tuning."""

import numpy as np
import pandas as pd

import f1.walkforward as W
from f1.data import DATA, load_results
from f1.features import build_features, pre_weekend_cols


def main():
    quotes = pd.read_csv(DATA / "parts/kalshi_prethu_2025.csv")
    covered = sorted(quotes.groupby("round").quoted.sum().loc[lambda n: n >= 3].index)
    assert len(covered) == 18
    assert not quotes.duplicated(["round", "code"]).any()
    X = build_features(load_results(), pd.read_csv(DATA / "racepace_ff1.csv"))
    F = pre_weekend_cols(X)
    W.MODELS["standings"] = lambda tr, te, f, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
    W.MODELS["logit"] = lambda tr, te, f, **kw: W.m_logit_pod(tr, te, F, C=0.03, hl=100)

    def shuffled(tr, te, f, **kw):
        t = tr.copy()
        t["podium"] = np.random.default_rng(0).permutation(t.podium.values)
        return W.m_logit_pod(t, te, F, C=0.03, hl=100)

    W.MODELS["null"] = shuffled
    R, P = W.walk_forward(X, [2025], ["standings", "logit", "null"],
                          ens={"ens": ["standings", "logit"]}, tie=None)
    entrants = X[(X.season == 2025) & X["round"].isin(covered)].copy()
    entrants = entrants.merge(quotes[["round", "code", "midpoint", "quoted"]],
                              on=["round", "code"], how="left", validate="one_to_one")
    quoted = entrants[entrants.quoted.fillna(False)].copy()
    market = []
    for r, g in entrants.groupby("round"):
        market.append({"round": r, "podium": W.metrics(g, -g.midpoint.values, tie=None)["podium"],
                       "probSum": float(g.midpoint.sum()), "entrants": len(g)})
    market = pd.DataFrame(market).set_index("round")
    strict = R[R["round"].isin(covered)].pivot(index="round", columns="model", values="podium")
    strict["market"] = market.podium
    print("strict overlap", strict[["market", "standings", "ens", "null"]].mean().round(6).to_dict())
    print("strict hits", (strict[["market", "standings", "ens", "null"]].sum() * 3).round().astype(int).to_dict())
    print("strict market minus standings", W.paired_boot(strict.market, strict.standings)[:2])
    print("strict market minus ens", W.paired_boot(strict.market, strict.ens)[:2])
    print("quote coverage", len(quoted), "of", len(entrants), "entrant rows across", len(covered), "races;")
    print("market probability sum per race", market.probSum.describe().round(3).to_dict())
    pred = P[P.model.isin(["logit", "null"]) & P["round"].isin(covered)]
    pred = pred.pivot(index=["round", "driver"], columns="model", values="score").reset_index()
    joined = quoted.merge(pred, on=["round", "driver"], validate="one_to_one")
    assert len(joined) == len(quoted) and joined[["midpoint", "logit", "null"]].notna().all().all()
    for model, prob in (("market", joined.midpoint), ("logit", -joined.logit), ("null", -joined.null)):
        p = np.clip(np.asarray(prob, float), 1e-6, 1 - 1e-6)
        y = joined.podium.values
        joined[model + "_brier"] = (p - y) ** 2
        joined[model + "_logloss"] = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    race = joined.groupby("round")[[f"{m}_{s}" for m in ("market", "logit", "null")
                                     for s in ("brier", "logloss")]].mean()
    for metric in ("brier", "logloss"):
        print(metric, "driver means", joined[[f"{m}_{metric}" for m in ("market", "logit", "null")]].mean().round(6).to_dict())
        print(metric, "race means", race[[f"{m}_{metric}" for m in ("market", "logit", "null")]].mean().round(6).to_dict())
        for ref in ("logit", "null"):
            print(metric, "market minus", ref, W.paired_boot(race[f"market_{metric}"], race[f"{ref}_{metric}"])[:2])


if __name__ == "__main__":
    main()
