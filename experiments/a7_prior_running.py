"""E120: does unusually short practice running in prior weekends predict attrition?"""

import glob

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import f1.walkforward as W
from experiments.common import table
from f1.data import DATA, load_results
from f1.features import pre_weekend_cols


PART = DATA / "parts/practice_short_2018_2025.csv"
RISK = ["d10_dnf", "t10_dnf", "d3_dnf", "t3_dnf", "era_round"]


def extract(X):
    if PART.exists():
        return pd.read_csv(PART)
    ff1 = pd.concat([pd.read_csv(f) for f in glob.glob(str(DATA / "parts/sessions_*.csv"))], ignore_index=True)
    ff1 = ff1[ff1.season <= 2022]
    openf1 = pd.read_csv(DATA / "laps.csv")
    openf1 = openf1[openf1.season.between(2023, 2025)]
    s = pd.concat([ff1, openf1], ignore_index=True)
    s = s[s.session.isin(["FP1", "FP2", "FP3"])].drop_duplicates(["season", "round", "session", "code"])
    n = s.groupby(["season", "round", "session"]).code.transform("nunique")
    s = s[n >= 15].copy()  # missing whole sessions are unknown, not zero running
    s["ratio"] = s.n / s.groupby(["season", "round", "session"]).n.transform("median")
    s["short"] = (s.ratio < 0.5).astype(float)
    entrants = X[["season", "round", "code", "team"]].drop_duplicates()
    s = s.merge(entrants, on=["season", "round", "code"], how="inner", validate="many_to_one")
    team = s.groupby(["season", "round", "team"]).agg(short=("short", "mean"),
                                                       sessions=("session", "nunique"),
                                                       driver_sessions=("code", "size")).reset_index()
    team.to_csv(PART, index=False)
    return team


def add_prior_running(X, source):
    keys = X[["season", "round", "k"]].drop_duplicates()
    p = source.merge(keys, on=["season", "round"], validate="many_to_one")
    out = []
    for k, cur in X.groupby("k", sort=True):
        past = p[(p.k < k) & (p.k >= k - 3)].copy()
        past["w"] = 0.5 ** ((k - past.k) / 10)
        g = (past.short * past.w).groupby(past.team).sum() / past.w.groupby(past.team).sum()
        f = cur[["season", "round", "code", "team"]].copy()
        f["prior_team_short"] = f.team.map(g)
        out.append(f[["season", "round", "code", "prior_team_short"]])
    return X.merge(pd.concat(out, ignore_index=True), on=["season", "round", "code"], validate="one_to_one")


def main():
    X = table(); X = X[X.season <= 2025].copy()
    source = extract(X)
    X = add_prior_running(X, source)
    results = load_results(); results = results[results.season <= 2025]
    X = X.merge(results[["season", "round", "code", "status"]], on=["season", "round", "code"], validate="one_to_one")
    X["dnf_y"] = (~(X.status.isin(["Finished", "Lapped"]) | X.status.str.startswith("+"))).astype(int)
    base = pre_weekend_cols(X)
    W.assert_pre_weekend(base + RISK + ["prior_team_short"])
    hazard = {}

    def risk(tr, te, cols, name):
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=500))
        pipe.fit(tr[cols], tr.dnf_y, logisticregression__sample_weight=W._w(tr, 100))
        p = pipe.predict_proba(te[cols])[:, 1]
        hazard[name, int(te.season.iloc[0]), int(te["round"].iloc[0])] = p
        return p

    W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
    W.MODELS["logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, base, C=0.03, hl=100)
    W.MODELS["logit_generic_survival"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, base, C=0.03, hl=100) * (1 - risk(tr, te, RISK, "generic"))
    W.MODELS["logit_short_survival"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, base, C=0.03, hl=100) * (1 - risk(tr, te, RISK + ["prior_team_short"], "short"))

    def shuffled(tr, te, F, **kw):
        t = tr.copy(); t["podium"] = np.random.default_rng(0).permutation(t.podium.values)
        return W.m_logit_pod(t, te, base, C=0.03, hl=100)

    W.MODELS["null_shuffled_labels"] = shuffled
    models = ["standings", "logit", "logit_generic_survival", "logit_short_survival", "null_shuffled_labels"]
    R, _ = W.walk_forward(X, range(2020, 2026), models,
                          ens={"ens": ["standings", "logit"],
                               "ens_generic": ["standings", "logit_generic_survival"],
                               "ens_short": ["standings", "logit_short_survival"]}, tie=None)
    R["split"] = np.where(R.season <= 2022, "inner", "outer")
    print("source coverage", source.groupby("season").agg(races=("round", "nunique"), team_rows=("team", "size")).to_string())
    print("target feature coverage", X[X.season.between(2020, 2025)].groupby("season").prior_team_short.apply(lambda s: s.notna().mean()).round(4).to_string())
    print("strict overlap", R.groupby(["model", "split"]).podium.mean().unstack().round(4).to_string())
    for split in ("inner", "outer"):
        z = R[R.split == split].pivot(index=["season", "round"], columns="model", values="podium")
        for candidate, ref in (("logit_short_survival", "logit_generic_survival"),
                               ("ens_short", "ens_generic"), ("ens_short", "ens"), ("ens_short", "standings")):
            diff, ci, _ = W.paired_boot(z[candidate], z[ref])
            print(split, "podium", candidate, "vs", ref, round(diff, 4), np.round(ci, 4).tolist())
        rows = []
        for (year, round_), g in X[X.season.isin(range(2020, 2023) if split == "inner" else range(2023, 2026))].groupby(["season", "round"]):
            for name in ("generic", "short"):
                p = hazard[name, year, round_]
                rows.extend((year, round_, name, (prob - label) ** 2,
                             -(label * np.log(prob) + (1 - label) * np.log1p(-prob)))
                            for prob, label in zip(p.clip(1e-6, 1 - 1e-6), g.dnf_y))
        h = pd.DataFrame(rows, columns=["season", "round", "model", "brier", "logloss"])
        print(split, "hazard mean", h.groupby("model")[["brier", "logloss"]].mean().round(5).to_string())
        q = h.groupby(["season", "round", "model"])[["brier", "logloss"]].mean().unstack()
        for metric in ("brier", "logloss"):
            diff, ci, _ = W.paired_boot(q[metric].short, q[metric].generic)
            print(split, "hazard", metric, "short minus generic", round(diff, 6), np.round(ci, 6).tolist())


if __name__ == "__main__":
    main()
