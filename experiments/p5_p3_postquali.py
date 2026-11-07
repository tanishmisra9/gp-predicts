"""E163 (retirement-aware re-selection) and E164 (sprint result feature), post-qualifying."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import f1.walkforward as W
from experiments.common import table
from run import COMPACT, baselines

REL = ["d10_dnf", "t10_dnf", "d3_dnf", "t3_dnf", "t10_order", "gridn"]


def p_fin(tr, te):
    t = tr; y = t.dnf.astype(int)
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.3, max_iter=500))
    m.fit(t[REL], y, logisticregression__sample_weight=W._w(t, 100)); return 1 - m.predict_proba(te[REL])[:, 1]


def main():
    X = table(); X = X[X.season <= 2025].copy()
    from f1.data import load_results
    r = load_results(); r["dnf"] = (~(r.status.isin(["Finished", "Lapped"]) | r.status.str.startswith("+"))).astype(float)
    X = X.merge(r[["season", "round", "code", "dnf"]], on=["season", "round", "code"], how="left")
    print("dnf col present:", "dnf" in X, " s_order present:", "s_order" in X, " sprint races:", X[X.s_order.notna()].groupby(["season", "round"]).ngroups if "s_order" in X else 0)
    baselines()
    def pod_x_fin(tr, te, F, top6=False, **kw):
        s = W.m_logit_pod(tr, te, COMPACT, C=0.3, hl=100); p = -s; f = p_fin(tr, te)
        if top6: rk = pd.Series(s).rank().values; f = np.where(rk <= 6, f, 1.0)
        return -(p * f)
    W.MODELS["logit_fin"] = lambda tr, te, F, **kw: pod_x_fin(tr, te, F)
    W.MODELS["logit_fin6"] = lambda tr, te, F, **kw: pod_x_fin(tr, te, F, top6=True)
    R, _ = W.walk_forward(X, range(2020, 2026), ["grid", "compact_logit", "logit_fin", "logit_fin6"], F=COMPACT)
    R["sp"] = np.where(R.season <= 2022, "inner", "outer")
    print("\nE163\n", R.groupby(["model", "sp"]).podium.mean().unstack().round(4).to_string())
    for sp in ("inner", "outer"):
        z = R[R.sp == sp].pivot(index=["season", "round"], columns="model", values="podium")
        for c in ("logit_fin", "logit_fin6"):
            d, ci, _ = W.paired_boot(z[c], z.compact_logit); print(f"{sp} {c} - compact_logit: {d:+.4f} [{ci[0]:+.3f},{ci[1]:+.3f}]")
    # E164
    X["s_gain"] = X.s_grid - X.s_order
    med = X.groupby(["season", "round"]).s_order.transform("median")
    X["s_order_f"] = X.s_order.fillna(med).fillna(10); X["s_gain_f"] = X.s_gain.fillna(0)
    W.MODELS["logit_sprint"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, COMPACT + ["s_order_f", "s_gain_f"], C=0.3, hl=100)
    R2, _ = W.walk_forward(X, range(2021, 2026), ["grid", "compact_logit", "logit_sprint"], F=COMPACT)
    spr = X[X.s_order.notna()][["season", "round"]].drop_duplicates()
    R2 = R2.merge(spr.assign(sprint=True), on=["season", "round"], how="left"); R2["sprint"] = R2.sprint.fillna(False)
    print("\nE164 sprint weekends:\n", R2[R2.sprint].groupby("model").podium.agg(["mean", "size"]).round(4).to_string())
    print("all races 2021-25:\n", R2.groupby("model").podium.mean().round(4).to_string())
    z = R2[R2.sprint].pivot(index=["season", "round"], columns="model", values="podium")
    d, ci, _ = W.paired_boot(z.logit_sprint, z.compact_logit); print(f"sprint races logit_sprint - compact_logit: {d:+.4f} [{ci[0]:+.3f},{ci[1]:+.3f}] n={len(z)}")
    d, ci, _ = W.paired_boot(z.logit_sprint, z.grid); print(f"sprint races logit_sprint - grid: {d:+.4f} [{ci[0]:+.3f},{ci[1]:+.3f}]")


if __name__ == "__main__":
    main()
