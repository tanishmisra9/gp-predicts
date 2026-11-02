"""Single entry point:  python run.py            -> full walk-forward evaluation (dev seasons + held-out 2026), writes results/
                       python run.py --dev      -> dev seasons only (never touches 2026; use while iterating)
All HTTP is cached (data/*.sqlite); delete data/results.csv + caches to refetch."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import f1.walkforward as W
from f1.data import load_results, DATA
from f1.features import build_features, pre_weekend_cols

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)

COMPACT = ["gridn", "log_grid", "q_gap", "team_qgap", "tm_qgap", "d10_order", "t10_order", "t3_order", "t10_q_gap",
           "d10_dnf", "t10_dnf", "t10_gain", "d_season_pts", "t_season_pts", "grid_x_corr", "grid_pen"]
DEV, TEST = [2023, 2024, 2025], [2026]


def baselines():
    W.MODELS["standings"] = lambda tr, te, F, **kw: -te.d_season_pts.fillna(0).values + 1e-3 * te.gridn.values  # championship order
    W.MODELS["prev_finish"] = lambda tr, te, F, **kw: te.prev_order.fillna(22).values
    W.MODELS["compact_logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, COMPACT, C=0.3, hl=100)


def report(R, P, X, seasons, final="compact_logit"):
    """Season-cycle analysis -> results/summary.md (also printed)."""
    L = []
    pr = lambda *a: (L.append(" ".join(str(x) for x in a)), print(*a))
    for s in seasons:
        Rs = R[R.season == s]
        pr(f"\n### Season {s} - aggregate over {Rs['round'].nunique()} rounds (mean per race)")
        pr(W.summarize(Rs).to_markdown())
        a = Rs[Rs.model == final].set_index("round").podium; b = Rs[Rs.model == "grid"].set_index("round").podium
        lo, hi = W.boot_ci(a.values); d = (a - b)
        dlo, dhi = W.boot_ci(d.values)
        pr(f"podium({final}) = {a.mean():.3f}  95% CI [{lo:.3f}, {hi:.3f}];  minus grid = {d.mean():+.3f}  CI [{dlo:+.3f}, {dhi:+.3f}]  (n={len(a)} races)")
        w = Rs[Rs.model == final].winner.values
        pr(f"winner({final}) = {w.mean():.3f}  95% CI [{W.boot_ci(w)[0]:.3f}, {W.boot_ci(w)[1]:.3f}]")
        pr(f"races with podium accuracy >= 75%: {(a >= .75).sum()}/{len(a)}")
    s = seasons[-1]
    Rs = R[(R.season == s) & R.model.isin([final, "grid"])]
    t = Rs.pivot(index="round", columns="model", values=["podium", "winner", "pairwise"]).round(3)
    t["cum_podium"] = t[("podium", final)].expanding().mean().round(3)
    t["cum_podium_grid"] = t[("podium", "grid")].expanding().mean().round(3)
    pr(f"\n### {s} per-round ({final} vs grid)"); pr(t.to_markdown())
    pr("\n### In-season data volume (accuracy by stage of the season, pooled over all evaluated seasons)")
    Rf = R[R.model.isin([final, "grid"])].assign(stage=lambda d: pd.cut(d["round"], [0, 5, 11, 30], labels=["R1-5", "R6-11", "R12+"]))
    pr(Rf.groupby(["stage", "model"], observed=True)[["podium", "winner", "pairwise"]].mean().round(3).unstack().to_markdown())
    (OUT / "summary.md").write_text("\n".join(L))


def oracle(pace, seasons):
    """DIAGNOSTIC UPPER BOUND, not a model: podium overlap if each driver's TRUE race pace in the target race were known."""
    d = load_results().merge(pace, on=["season", "round", "code"], how="left")
    d = d[d.season.isin(seasons)]
    ov = lambda g: len(set(g.nsmallest(3, "pace_gap").driver) & set(g.nsmallest(3, "order").driver)) / 3
    return d.assign(pace_gap=d.pace_gap.fillna(99)).groupby(["season", "round"]).apply(ov, include_groups=False)


def main_pre(seasons=None):
    """Pre-weekend setting: NO data from the target weekend (no grid/quali/sprint/practice)."""
    seasons = seasons or DEV + TEST
    df = load_results(); pace = pd.read_csv(DATA / "racepace_ff1.csv")
    X = build_features(df, pace); PRE = pre_weekend_cols(X)
    standings = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
    W.MODELS.update(
        prev_podium=lambda tr, te, F, **kw: te.prev_order.fillna(22).values,
        standings=standings,
        form_d3=lambda tr, te, F, **kw: te.d3_order.fillna(22).values,
        pre_logit=lambda tr, te, F, **kw: W.m_logit_pod(tr, te, PRE, C=0.03, hl=100),
        pre_lgb=lambda tr, te, F, **kw: W.m_lgb_pod(tr, te, PRE, hl=100))
    models = ["prev_podium", "standings", "form_d3", "pre_logit", "pre_lgb"]
    R, P = W.walk_forward(X, seasons, models, ens={"ens": ["standings", "pre_logit"]})
    R.to_csv(OUT / "preweekend_per_round.csv", index=False)
    orc = oracle(pace, seasons)
    for s in seasons:
        Rs = R[R.season == s]
        print(f"\n### PRE-WEEKEND season {s} ({Rs['round'].nunique()} races): podium accuracy per model + oracle ceiling")
        print(W.summarize(Rs)[["winner", "podium", "top10", "pairwise", "mae"]].assign(
            oracle_podium=np.nan).to_markdown())
        print(f"oracle (true race pace known; diagnostic upper bound): podium {orc.loc[s].mean():.3f}")
    return R, X


def main(dev_only=False):
    baselines()
    df = load_results()
    laps = pd.read_csv(DATA / "laps.csv") if (DATA / "laps.csv").exists() else None
    X = build_features(df, laps)
    models = ["grid", "standings", "prev_finish", "compact_logit"]
    seasons = DEV if dev_only else DEV + TEST
    R, P = W.walk_forward(X, seasons, models)
    R.to_csv(OUT / "per_round.csv", index=False)
    P.to_csv(OUT / "predictions.csv", index=False)
    report(R, P, X, seasons)
    return R, X


if __name__ == "__main__":
    if "--pre" in sys.argv:
        main_pre()
    else:
        main(dev_only="--dev" in sys.argv)
