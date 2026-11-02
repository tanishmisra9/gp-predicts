"""A4 pilot: does past cause-specific failure exposure improve next-race predictions?"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import f1.walkforward as W
from experiments.common import table
from f1.data import load_results
from f1.features import pre_weekend_cols

# Only explicit mechanical statuses. "Retired" is unknown, not a known negative cause.
MECH = {"Engine", "Power Unit", "Power loss", "Electrical", "Electronics", "Turbo",
        "Exhaust", "Overheating", "Fuel pressure", "Fuel pump", "Fuel leak",
        "Oil leak", "Water pressure", "Water leak", "Water pump", "Cooling system",
        "Radiator", "Gearbox", "Hydraulics", "Transmission", "Differential",
        "Driveshaft", "Clutch", "Mechanical", "Brakes", "Suspension"}
UNKNOWN = {"Retired", "Did not start", "Withdrew", "Disqualified", "Vibrations", "Damage"}
NEW = ["mech_d_rate", "mech_t_rate", "mech_t_last3"]


def add_mechanical_history(X, results):
    """Inputs: dated results and feature rows. Output: past-only cause rates for every row."""
    k = X.drop_duplicates(["season", "round"])[["season", "round", "k"]]
    r = results.merge(k, on=["season", "round"], validate="many_to_one")
    r["mech"] = r.status.isin(MECH).astype(float)
    rows = []
    for race_k, cur in X.groupby("k", sort=True):
        past = r[r.k < race_k].copy()
        f = cur[["season", "round", "code", "team"]].copy()
        if len(past):
            past["w"] = 0.5 ** ((race_k - past.k) / 10)
            for key, out, prior_laps in (("code", "mech_d_rate", 1000), ("team", "mech_t_rate", 2000)):
                g = past.groupby(key).apply(
                    lambda p: (1000 * ((p.mech * p.w).sum() + prior_laps * 0.0005) /
                               ((p.laps * p.w).sum() + prior_laps)), include_groups=False)
                f[out] = f[key].map(g).fillna(0.5)
            f["mech_t_last3"] = f.team.map(past[past.k >= race_k - 3].groupby("team").mech.max()).fillna(0)
        else:
            f["mech_d_rate"] = f["mech_t_rate"] = 0.5
            f["mech_t_last3"] = 0.0
        rows.append(f[["season", "round", "code"] + NEW])
    out = X.merge(pd.concat(rows, ignore_index=True), on=["season", "round", "code"], validate="one_to_one")
    out = out.merge(results[["season", "round", "code", "status"]], on=["season", "round", "code"], validate="one_to_one")
    out["mech_y"] = out.status.isin(MECH).astype(float)
    return out


def main():
    # Deliberately exclude 2026 from this development experiment.
    X = table(); X = X[X.season <= 2025].copy()
    results = load_results(); results = results[results.season <= 2025].copy()
    X = add_mechanical_history(X, results)
    print("Named mechanical and unknown-cause status counts by season:")
    print(X.assign(unknown=X.status.isin(UNKNOWN)).groupby("season")[["mech_y", "unknown"]].sum().astype(int).to_string())
    base = pre_weekend_cols(X)
    risk_base = ["d10_dnf", "t10_dnf", "d3_dnf", "t3_dnf", "era_round"]
    W.assert_pre_weekend(base + risk_base + NEW)
    hazards = {}

    def risk(tr, te, cols, name):
        # Unknown causes are excluded from hazard training and scoring.
        t = tr[~tr.status.isin(UNKNOWN)]
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.1, max_iter=500))
        pipe.fit(t[cols], t.mech_y, logisticregression__sample_weight=W._w(t, 100))
        p = pipe.predict_proba(te[cols])[:, 1]
        hazards[(name, int(te.season.iloc[0]), int(te["round"].iloc[0]))] = p
        return p

    W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
    W.MODELS["logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, base, C=0.03, hl=100)
    W.MODELS["logit_x_generic_risk"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, base, C=0.03, hl=100) * (1 - risk(tr, te, risk_base, "generic"))
    W.MODELS["logit_x_cause_risk"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, base, C=0.03, hl=100) * (1 - risk(tr, te, risk_base + NEW, "cause"))
    def shuffled(tr, te, F, **kw):
        t = tr.copy()
        t["podium"] = np.random.default_rng(0).permutation(t.podium.values)
        return W.m_logit_pod(t, te, base, C=0.03, hl=100)
    W.MODELS["null_shuffled_labels"] = shuffled
    models = ["standings", "logit", "logit_x_generic_risk", "logit_x_cause_risk", "null_shuffled_labels"]
    R, _ = W.walk_forward(X, [2020, 2021, 2022, 2023, 2024, 2025], models,
                          ens={"ens": ["standings", "logit"]}, tie=None)
    R["split"] = np.where(R.season <= 2022, "inner", "outer")
    print("Strict podium overlap (same races):")
    print(R.groupby(["model", "split"]).podium.mean().unstack().round(4).to_string())
    for split, years in (("inner", [2020, 2021, 2022]), ("outer", [2023, 2024, 2025])):
        d = R[R.season.isin(years)].pivot(index=["season", "round"], columns="model", values="podium")
        for name in ("logit_x_generic_risk", "logit_x_cause_risk"):
            diff, ci, _ = W.paired_boot(d[name], d.standings)
            print(split, name, "vs standings", round(diff, 4), np.round(ci, 4).tolist())
        eligible = X[X.season.isin(years) & ~X.status.isin(UNKNOWN)]
        for name in ("generic", "cause"):
            scores = []
            for (_, _), g in eligible.groupby(["season", "round"], sort=True):
                key = (name, int(g.season.iloc[0]), int(g["round"].iloc[0]))
                full = X[(X.season == key[1]) & (X["round"] == key[2])]
                scores.extend(zip(g.mech_y, pd.Series(hazards[key], index=full.index).loc[g.index]))
            y, p = np.array(scores).T
            if y.sum() == 0:
                print(split, name, "mechanical hazard not estimable: zero named events")
            else:
                print(split, name, "mechanical events", int(y.sum()), "n", len(y),
                      "Brier", round(brier_score_loss(y, p), 5),
                      "logloss", round(log_loss(y, p, labels=[0, 1]), 5))


if __name__ == "__main__":
    main()
