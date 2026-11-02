"""Predict a race from data available before lights-out.
  python predict.py                 -> next race after the last completed one (needs its qualifying on Jolpica)
  python predict.py 2026 19         -> "as-of" replay of a finished race: outcomes hidden from features/training, actual shown after
Uses the frozen final model (run.COMPACT logit). Grid = official grid if the race row exists, else qualifying order (penalties unknown)."""
import sys
import warnings

import numpy as np
import pandas as pd

import run
import f1.walkforward as W
from f1.data import _pages, _num, _qsec, load_results
from f1.features import build_features

warnings.filterwarnings("ignore")
OUTCOMES = ["finish", "order", "status", "points", "laps"]


def stub_from_quali(season, rnd):
    races = [r for r in _pages(f"{season}/{rnd}/qualifying") if (int(r["season"]), int(r["round"])) == (season, rnd)]
    if not races or not races[0].get("QualifyingResults"):
        return None
    r = races[0]
    rows = [dict(season=season, round=rnd, date=r["date"], race=r["raceName"], circuit=r["Circuit"]["circuitId"],
                 driver=x["Driver"]["driverId"], code=x["Driver"].get("code"), team=x["Constructor"]["constructorId"],
                 grid=int(x["position"]), finish=np.nan, order=np.nan, status="", points=0.0, laps=0,
                 q_pos=_num(x["position"]), q1=_qsec(x.get("Q1")), q2=_qsec(x.get("Q2")), q3=_qsec(x.get("Q3")),
                 s_grid=np.nan, s_order=np.nan, s_points=np.nan) for x in r["QualifyingResults"]]
    return pd.DataFrame(rows)


def main(season=None, rnd=None):
    df = load_results(); df["date"] = pd.to_datetime(df["date"])
    if season is None:
        season, rnd = (int(v) for v in df.sort_values(["season", "round"]).iloc[-1][["season", "round"]]); rnd += 1
    actual = None
    cur = df[(df.season == season) & (df["round"] == rnd)]
    if len(cur):  # as-of replay: keep the pre-race columns (incl. official grid), hide outcomes
        actual = cur.sort_values("order")[["driver", "order"]].head(3)
        cur = cur.copy(); cur[OUTCOMES] = np.nan; cur["status"] = ""
    else:
        cur = stub_from_quali(season, rnd)
        if cur is None:
            sys.exit(f"No qualifying data for {season} round {rnd} yet - run again after qualifying.")
        print("(grid = qualifying order; grid penalties are not known to the data source)")
    hist = df[(df.season < season) | ((df.season == season) & (df["round"] < rnd))]
    X = build_features(pd.concat([hist, cur], ignore_index=True))
    te = X[(X.season == season) & (X["round"] == rnd)]
    tr = X[(X.k < te.k.iloc[0]) & (X.season >= 2019)]
    p = -W.m_logit_pod(tr, te, run.COMPACT, C=0.3, hl=100)
    out = te[["driver", "team", "gridn"]].assign(p_podium=p.round(3)).sort_values("p_podium", ascending=False)
    out["pred_pos"] = range(1, len(out) + 1)
    print(f"\n{season} round {rnd}: predicted order (trained on {tr.k.nunique()} earlier races)")
    print(out.to_string(index=False))
    print("\nPredicted podium:", ", ".join(out.driver.head(3)))
    if actual is not None:
        print("Actual podium:   ", ", ".join(actual.driver), "(replay only; not seen by the model)")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*(int(x) for x in a)) if a else main()
