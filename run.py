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
from f1.features import build_features

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)

COMPACT = ["gridn", "log_grid", "q_gap", "team_qgap", "tm_qgap", "d10_order", "t10_order", "t3_order", "t10_q_gap",
           "d10_dnf", "t10_dnf", "t10_gain", "d_season_pts", "t_season_pts", "grid_x_corr", "grid_pen"]
DEV, TEST = [2023, 2024, 2025], [2026]


def baselines():
    W.MODELS["standings"] = lambda tr, te, F, **kw: -te.d_season_pts.fillna(0).values + 1e-3 * te.gridn.values  # championship order
    W.MODELS["prev_finish"] = lambda tr, te, F, **kw: te.prev_order.fillna(22).values
    W.MODELS["compact_logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, COMPACT, C=0.3, hl=100)


def main(dev_only=False):
    baselines()
    df = load_results()
    laps = pd.read_csv(DATA / "laps.csv") if (DATA / "laps.csv").exists() else None
    X = build_features(df, laps)
    models = ["grid", "standings", "prev_finish", "compact_logit"]
    seasons = DEV if dev_only else DEV + TEST
    R, P = W.walk_forward(X, seasons, models)
    R.to_csv(OUT / "per_round.csv", index=False)
    for s in seasons:
        print(f"\n=== season {s} ===\n", W.summarize(R[R.season == s]).to_string())
    return R, X


if __name__ == "__main__":
    main(dev_only="--dev" in sys.argv)
