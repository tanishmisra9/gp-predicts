"""Forecast grid may train on past grids, never read the target grid."""

import numpy as np

from experiments.b8_grid_forecast import grid_forecast
from experiments.b8_grid_feature import add_prior_grid
from experiments.common import table
from f1.features import pre_weekend_cols
import f1.walkforward as W


def test_grid_forecast_ignores_target_grid():
    X = table(); X = X[X.season <= 2025].copy()
    features = pre_weekend_cols(X)
    W.assert_pre_weekend(features)
    for year, round_ in ((2023, 7), (2025, 20)):
        te = X[(X.season == year) & (X["round"] == round_)]
        tr = X[(X.k < te.k.iloc[0]) & (X.season >= 2019)]
        original = grid_forecast(tr, te, features)
        altered = te.copy()
        altered["gridn"] = np.arange(len(te), 0, -1)
        altered["q_pos"] = np.arange(len(te), 0, -1)
        np.testing.assert_allclose(original, grid_forecast(tr, altered, features), rtol=0, atol=0)
    changed = tr.copy()
    changed["gridn"] = 21
    assert not np.allclose(original, grid_forecast(changed, te, features))


def test_out_of_fold_grid_feature_ignores_target_grid():
    X = table()
    X = X[(X.season < 2023) | ((X.season == 2023) & (X["round"] <= 7))].copy()
    target = (X.season == 2023) & (X["round"] == 7)
    original = add_prior_grid(X)
    changed = X.copy()
    changed.loc[target, "gridn"] = 21
    got = add_prior_grid(changed)
    np.testing.assert_allclose(original.loc[target, "prior_grid"], got.loc[target, "prior_grid"], rtol=0, atol=0)
    prior = (X.season == 2023) & (X["round"] == 6)
    changed = X.copy()
    changed.loc[prior, "gridn"] = 21
    got = add_prior_grid(changed)
    assert not np.allclose(original.loc[target, "prior_grid"], got.loc[target, "prior_grid"])


if __name__ == "__main__":
    test_grid_forecast_ignores_target_grid()
    test_out_of_fold_grid_feature_ignores_target_grid()
    print("grid forecast leak test passed")
