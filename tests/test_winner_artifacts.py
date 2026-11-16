"""Recompute winner/podium hits from saved rankings, independently of model training."""
import numpy as np
import pandas as pd


def test_saved_rankings():
    truth = pd.read_csv("data/results.csv").loc[lambda d: d.season.between(2020, 2025), ["season", "round", "driver", "order"]]
    for experiment in (169, 170, 171, 172, 173, 175, 176, 177, 178, "178_inner"):
        P = pd.read_csv(f"results/e{experiment}_predictions.csv", float_precision="round_trip")
        R = pd.read_csv(f"results/e{experiment}_per_race.csv")
        assert P.season.max() == (2022 if experiment == "178_inner" else 2025)
        assert not P.duplicated(["season", "round", "model", "driver"]).any()
        merged = P.merge(truth, on=["season", "round", "driver"], validate="many_to_one")
        merged = merged.sort_values(["season", "round", "model", "score", "driver"])
        merged["pred_rank"] = merged.groupby(["season", "round", "model"]).cumcount() + 1
        merged["winner_hit"] = merged.pred_rank.eq(1) & merged.order.eq(1)
        merged["podium_hit"] = merged.pred_rank.le(3) & merged.order.le(3)
        actual = merged.groupby(["season", "round", "model"])[["winner_hit", "podium_hit"]].sum()
        expected = R[R.model.isin(P.model.unique())].set_index(["season", "round", "model"]).sort_index()
        assert len(actual) == len(expected)
        np.testing.assert_array_equal(actual.winner_hit, expected.winner)
        np.testing.assert_allclose(actual.podium_hit / 3, expected.podium, atol=1e-12)


if __name__ == "__main__":
    test_saved_rankings()
    print("all tested saved winner and podium rankings reproduce the logged metrics")
