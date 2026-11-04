"""E143: unchanged 2023 X/Y team profiles against 2024 team race pace."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import f1.walkforward as W
from experiments.a1_2025_team_transfer import evaluate, extract
from f1.data import DATA, load_results


SESSIONS = {
    "bahrain": 7953, "jeddah": 7779, "albert_park": 7787, "baku": 9070,
    "miami": 9078, "monaco": 9094, "catalunya": 9102, "villeneuve": 9110,
    "red_bull_ring": 9118, "silverstone": 9126, "hungaroring": 9133,
    "spa": 9141, "zandvoort": 9149, "monza": 9157,
    "marina_bay": 9165, "suzuka": 9173, "losail": 9221,
    "americas": 9213, "rodriguez": 9181, "interlagos": 9205,
    "vegas": 9189, "yas_marina": 9197,
}
PART = DATA / "parts/race_xy_multistint_all_2023.csv"


def main():
    source = extract(SESSIONS, PART, source_season=2023)
    print("profile driver-races by circuit", source.groupby("circuit").code.nunique().reindex(SESSIONS).to_dict())
    results = load_results().query("season <= 2024").copy()
    # These constructors continued under new names in 2024.
    old = results.season.eq(2023)
    results.loc[old, "team"] = results.loc[old, "team"].replace({"alfa": "sauber", "alphatauri": "rb"})
    pace = pd.read_csv(DATA / "racepace_ff1.csv")
    r, groups = evaluate(source, results, pace, SESSIONS, source_season=2023)
    print("2024 team pace Spearman", r.round(4).to_string())
    print("mean", r[["standings", "profile", "prior_pace", "blend"]].mean().round(4).to_dict())
    defined = r.standings.notna()
    for name in ("profile", "prior_pace", "blend"):
        print(name, "minus standings paired on defined races", defined.sum(),
              W.paired_boot(r.loc[defined, name], r.loc[defined, "standings"])[:2])
    rng = np.random.default_rng(143)
    null = np.array([np.mean([spearmanr(g.profile_score, rng.permutation(g.pace_gap)).statistic
                              for g in groups]) for _ in range(5000)])
    print("profile shuffled-pace null mean", null.mean(),
          "one-sided p", (1 + (null >= r.profile.mean()).sum()) / (len(null) + 1))


if __name__ == "__main__":
    main()
