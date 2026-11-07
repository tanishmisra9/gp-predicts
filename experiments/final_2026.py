"""E159: the single final pre-weekend 2026 look. Frozen candidate = ens + E156 rounds-1-3 testing rule."""
import glob
import numpy as np
import pandas as pd
import f1.walkforward as W
from experiments.common import table
from f1.data import load_results
from f1.features import pre_weekend_cols


def main():
    X = table(); res = load_results()
    base = pre_weekend_cols(X)
    tst = pd.concat([pd.read_csv(f) for f in glob.glob("data/parts/testing_*.csv")]); tst = tst[tst.season == 2026]
    r1 = res[(res["round"] == 1) & (res.season == 2026)][["season", "code", "team"]]
    tt = (tst.groupby(["season", "code"])[["best_gap", "lr_gap"]].mean().reset_index().merge(r1, on=["season", "code"])
          .groupby(["season", "team"])[["best_gap", "lr_gap"]].mean().add_prefix("tt_").reset_index())
    print("2026 testing teams covered:", len(tt))
    W.MODELS["standings"] = lambda tr, te, f, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
    W.MODELS["logit"] = lambda tr, te, f, **kw: W.m_logit_pod(tr, te, base, C=0.03, hl=100)

    def shuffled(tr, te, f, **kw):
        t = tr.copy(); t["podium"] = np.random.default_rng(0).permutation(t.podium.values)
        return W.m_logit_pod(t, te, base, C=0.03, hl=100)
    W.MODELS["null"] = shuffled
    R, P = W.walk_forward(X, [2026], ["standings", "logit", "null"], ens={"ens": ["standings", "logit"]}, tie=None)
    Xt = X[X.season == 2026].merge(tt, on=["season", "team"], how="left")
    ens = P[P.model == "ens"].rename(columns={"score": "ens"})[["season", "round", "driver", "ens"]]
    rows = []
    for (s, r), g in Xt.merge(ens, on=["season", "round", "driver"]).groupby(["season", "round"]):
        sc = g.ens.values
        if r <= 3 and g.tt_best_gap.notna().any():
            ts = (g.tt_lr_gap.rank(pct=True).fillna(.5) + g.tt_best_gap.rank(pct=True).fillna(.5)) / 2
            sc = 0.75 * g.ens.values + 0.25 * ts.values
        m = W.metrics(g, sc, None); rows.append(dict(season=s, round=r, model="candidate", **m))
    R = pd.concat([R, pd.DataFrame(rows)], ignore_index=True)
    S = W.summarize(R); print(S.to_string())
    z = R.pivot(index=["season", "round"], columns="model", values="podium")
    print("n races", len(z))
    for a, b in (("candidate", "ens"), ("candidate", "standings"), ("ens", "standings")):
        print(a, "minus", b, W.paired_boot(z[a], z[b])[:2])
    print("candidate >=2 of 3:", (z.candidate >= 2 / 3 - 1e-9).mean().round(3), " races >=0.75:", int((z.candidate >= 0.75).sum()))
    print("per round candidate:", z.candidate.round(2).tolist())
    # re-scope (b): podium Brier / log-loss, logit vs standings-rank prior (prior = empirical podium rate by standings rank from training rows)
    pr = P[P.model == "logit"].merge(X[["season", "round", "driver", "podium", "d_season_pts"]], on=["season", "round", "driver"])
    p = (-pr.score).clip(1e-6, 1 - 1e-6)
    hist = X[X.season < 2026].copy(); hist["rk"] = hist.groupby(["season", "round"]).d_season_pts.rank(ascending=False, method="first")
    rate = hist.groupby("rk").podium.mean()
    pr["rk"] = pr.groupby(["season", "round"]).d_season_pts.rank(ascending=False, method="first")
    q = pr.rk.map(rate).fillna(rate.min()).clip(1e-6, 1 - 1e-6)
    y = pr.podium.values
    for name, v in (("logit", p), ("standings-rank prior", q)):
        print(f"{name}: Brier {np.mean((v - y) ** 2):.5f}  logloss {-np.mean(y * np.log(v) + (1 - y) * np.log(1 - v)):.5f}")
    d = ((p - y) ** 2 - (q - y) ** 2); print("Brier logit-prior race-paired", W.paired_boot(d.groupby([pr.season, pr["round"]]).mean(), 0 * d.groupby([pr.season, pr["round"]]).mean())[:2])


if __name__ == "__main__":
    main()
