"""E178: choose winner regularization on inner races, then lock it before outer fitting."""
import json
from functools import partial
from pathlib import Path

import pandas as pd

import f1.walkforward as W
from experiments.e168_winner import win_score
from f1.data import DATA, load_results
from f1.features import build_features, pre_weekend_cols

C_GRID = (.003, .03, .3, 3.)


def select_regularization(R):
    assert set(R.season).issubset({2020, 2021, 2022}), "outer rows must never select C"
    assert not R.duplicated(["season", "round", "model"]).any()
    pivot = R.pivot(index=["season", "round"], columns="model", values="winner")
    assert set(pivot.columns) == {f"C_{c:g}" for c in C_GRID}
    assert pivot.notna().all().all()
    hits = pivot.sum()
    return min(C_GRID, key=lambda c: (-hits[f"C_{c:g}"], c))


def main():
    df = load_results().loc[lambda d: d.season <= 2025]
    pace = pd.read_csv(DATA / "racepace_ff1.csv").loc[lambda d: d.season <= 2025]
    X = build_features(df, pace); F = pre_weekend_cols(X)
    W.assert_pre_weekend(F)
    for c in C_GRID:
        W.MODELS[f"C_{c:g}"] = partial(win_score, C=c)
    inner, predictions = W.walk_forward(X, [2020, 2021, 2022], [f"C_{c:g}" for c in C_GRID], F=F, tie=None)
    inner.to_csv("results/e178_inner_per_race.csv", index=False)
    predictions.to_csv("results/e178_inner_predictions.csv", index=False)
    selected = select_regularization(inner)
    selection = dict(C_grid=C_GRID, selected_C=selected, inner_seasons=[2020, 2021, 2022],
                     inner_winner_hits=inner.groupby("model").winner.sum().to_dict(),
                     tie_rule="highest inner winner count, then smallest C", outer_run=selected != .03)
    Path("results/e178_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print("INNER, optimized selection data", inner.groupby("model")[["winner", "podium"]].agg(["sum", "mean"]).to_string(), flush=True)
    print(inner.pivot_table(index="season", columns="model", values="winner", aggfunc="sum").to_string(), flush=True)
    print("LOCKED SELECTION", selection, flush=True)
    if selected == .03:
        print("STOP: existing C=.03 selected; no new outer evaluation", flush=True)
        return
    W.MODELS.update(
        selected_win=partial(win_score, C=selected),
        selected_null=partial(win_score, C=selected, shuffled=True),
        win_logit=win_score,
        standings=lambda tr, te, F: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).to_numpy(),
        pre_logit=lambda tr, te, F: W.m_logit_pod(tr, te, F, C=.03, hl=100),
    )
    R, P = W.walk_forward(X, [2023, 2024, 2025], ["selected_win", "selected_null", "win_logit", "standings", "pre_logit"],
                          F=F, ens={"ens": ["standings", "pre_logit"]}, tie=None)
    print("OUTER, locked candidate", R.groupby("model")[["winner", "podium"]].agg(["sum", "mean"]).to_string(), flush=True)
    print(R.pivot_table(index="season", columns="model", values="winner", aggfunc="sum").to_string(), flush=True)
    a = R[R.model.eq("selected_win")].set_index(["season", "round"]).winner
    for reference in ("standings", "ens", "win_logit"):
        b = R[R.model.eq(reference)].set_index(["season", "round"]).winner
        print("selected_win minus", reference, W.paired_boot(a, b), flush=True)
    R.to_csv("results/e178_per_race.csv", index=False)
    P.to_csv("results/e178_predictions.csv", index=False)


if __name__ == "__main__":
    main()
