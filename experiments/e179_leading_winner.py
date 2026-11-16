"""E179: strictly past laps-led share as a winner-specific input."""
import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.e168_winner import win_score
from f1.data import DATA, load_results
from f1.features import build_features, pre_weekend_cols

START_FILES = [DATA / "parts" / name for name in ("start_2018_2021.csv", "start_2022_2024.csv", "start_2025_2026.csv")]
LEAD_FEATURES = [f"prior_{who}_lead{hl}" for who in ("driver", "team") for hl in (3, 10)]


def add_leading_history(X, events):
    source = X[["season", "round", "k", "code", "team", "era"]].merge(
        events[["season", "round", "code", "led_share"]], on=["season", "round", "code"], how="left", validate="one_to_one")
    teams = source.groupby(["k", "team"], sort=True).agg(
        era=("era", "first"), led_share=("led_share", lambda s: s.sum(min_count=1))).reset_index()
    out = X.copy()
    for k, cur in X.groupby("k", sort=True):
        for who, key, table, em in (("driver", "code", source, .3), ("team", "team", teams, .1)):
            past = table[table.k.lt(k)].dropna(subset=["led_share"])
            for hl in (3, 10):
                w = .5 ** ((k - past.k) / hl) * np.where(past.era < cur.era.iloc[0], em, 1.)
                rate = (w * past.led_share).groupby(past[key]).sum() / w.groupby(past[key]).sum()
                out.loc[cur.index, f"prior_{who}_lead{hl}"] = cur[key].map(rate)
    return out


def main():
    df = load_results().loc[lambda d: d.season <= 2025]
    pace = pd.read_csv(DATA / "racepace_ff1.csv").loc[lambda d: d.season <= 2025]
    X = build_features(df, pace); base = pre_weekend_cols(X)
    events = pd.concat([pd.read_csv(p) for p in START_FILES], ignore_index=True).loc[lambda d: d.season <= 2025]
    X = add_leading_history(X, events); augmented = base + LEAD_FEATURES
    W.assert_pre_weekend(augmented)
    W.MODELS.update(
        standings=lambda tr, te, F: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).to_numpy(),
        pre_logit=lambda tr, te, F: W.m_logit_pod(tr, te, base, C=.03, hl=100),
        win_logit=lambda tr, te, F: win_score(tr, te, base),
        leading_win=lambda tr, te, F: win_score(tr, te, augmented),
        leading_null=lambda tr, te, F: win_score(tr, te, augmented, shuffled=True),
    )
    R, P = W.walk_forward(X, list(range(2020, 2026)), ["standings", "pre_logit", "win_logit", "leading_win", "leading_null"],
                          F=augmented, ens={"ens": ["standings", "pre_logit"]}, tie=None)
    for split, years in (("inner", [2020, 2021, 2022]), ("outer", [2023, 2024, 2025])):
        r = R[R.season.isin(years)]
        print(split, r.groupby("model")[["winner", "podium"]].agg(["sum", "mean"]).round(4).to_string(), flush=True)
        print(r.pivot_table(index="season", columns="model", values="winner", aggfunc="sum").astype(int).to_string(), flush=True)
        print("feature coverage", X[X.season.isin(years)][LEAD_FEATURES].notna().mean().to_dict(), flush=True)
        a = r[r.model.eq("leading_win")].set_index(["season", "round"]).winner
        for ref in ("standings", "ens", "win_logit"):
            b = r[r.model.eq(ref)].set_index(["season", "round"]).winner
            print("leading_win minus", ref, W.paired_boot(a, b), flush=True)
    R.to_csv("results/e179_per_race.csv", index=False)
    P.to_csv("results/e179_predictions.csv", index=False)


if __name__ == "__main__":
    main()
