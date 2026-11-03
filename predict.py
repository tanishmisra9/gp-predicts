"""Predict the podium of a race.
  python predict.py              -> NEXT race, using ONLY earlier races (no practice/quali/grid of that weekend). Default.
  python predict.py 2026 19      -> as-of replay of a finished race (everything about that weekend hidden), actual podium shown after
  python predict.py --post [S R] -> post-qualifying variant (grid-aware, ~65% podium); needs the weekend's qualifying on Jolpica
Pre-weekend model = rank-average of (championship-standings order) and a pace-aware logistic podium model (see EXPERIMENTS.md E8)."""
import sys
import warnings

import numpy as np
import pandas as pd

import f1.walkforward as W
from f1.data import DATA, _pages, load_results
from f1.features import build_features, pre_weekend_cols

warnings.filterwarnings("ignore")
WEEKEND = ["grid", "q_pos", "q1", "q2", "q3", "s_grid", "s_order", "s_points", "finish", "order", "points", "laps"]


def main(season=None, rnd=None, post=False):
    if post:
        import predict_post
        return predict_post.main(season, rnd)
    df = load_results(); df["date"] = pd.to_datetime(df["date"])
    last = df.sort_values(["season", "round"]).iloc[-1]
    if season is None:
        season, rnd = int(last.season), int(last["round"]) + 1
    cur = df[(df.season == season) & (df["round"] == rnd)]
    actual = None
    if len(cur):  # as-of replay of a finished race: wipe everything about the weekend except the entry list
        actual = cur.sort_values("order").driver.head(3).tolist()
        cur = cur.copy(); cur[WEEKEND] = np.nan; cur["status"] = ""
    else:  # upcoming race: entry list = last race's drivers/teams (assumption: no line-up change); circuit from the calendar
        info = [r for r in _pages(f"{season}/{rnd}") if int(r["round"]) == rnd]
        if not info:
            sys.exit(f"{season} round {rnd} is not on the calendar.")
        prev = df[(df.season == last.season) & (df["round"] == last["round"])]
        cur = prev.copy(); cur[WEEKEND] = np.nan; cur["status"] = ""
        cur["season"], cur["round"], cur["date"] = season, rnd, info[0]["date"]
        cur["race"], cur["circuit"] = info[0]["raceName"], info[0]["Circuit"]["circuitId"]
        print(f"(entry list assumed = {last.race} line-up; target: {info[0]['raceName']} {info[0]['date']})")
    hist = df[(df.season < season) | ((df.season == season) & (df["round"] < rnd))]
    pace = pd.read_csv(DATA / "racepace_ff1.csv")
    X = build_features(pd.concat([hist, cur], ignore_index=True), pace); PRE = pre_weekend_cols(X)
    W.assert_pre_weekend(PRE)
    te = X[(X.season == season) & (X["round"] == rnd)]; tr = X[(X.k < te.k.iloc[0]) & (X.season >= 2019)]
    st = (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
    lg = W.m_logit_pod(tr, te, PRE, C=0.03, hl=100)
    score = (pd.Series(st).rank().values + pd.Series(lg).rank().values) / 2
    out = te[["driver", "team"]].assign(score=score).sort_values("score")
    out["pred_pos"] = range(1, len(out) + 1)
    print(f"\n{season} round {rnd}: PRE-WEEKEND prediction (trained on {tr.k.nunique()} earlier races; no data from this weekend)")
    print(out[["pred_pos", "driver", "team"]].head(10).to_string(index=False))
    print("\nPredicted podium:", ", ".join(out.driver.head(3)))
    standing = te[["driver"]].assign(score=st).sort_values(["score", "driver"]).driver.head(3)
    print("Standings-only podium:", ", ".join(standing))
    if actual:
        print("Actual podium:   ", ", ".join(actual), "(replay only; hidden from the model)")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--post"]
    main(*(int(x) for x in a), post="--post" in sys.argv)
