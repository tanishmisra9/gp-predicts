"""A3 driver-style forecast must ignore target and later stint summaries."""
import pandas as pd

from experiments.a3_tyre_degradation import PART, prior_style
from experiments.common import table


def test_style_history_uses_only_past():
    X = pd.DataFrame([(2022, 1, 0, "AAA"), (2022, 2, 1, "AAA"),
                      (2022, 3, 2, "AAA"), (2022, 4, 3, "AAA")],
                     columns=["season", "round", "k", "code"])
    style = X[["season", "round", "code"]].assign(style=[0.1, 0.2, None, 0.4])
    base = prior_style(X, style)
    assert pd.isna(base.loc[0, "prior_style"])
    assert base.loc[1, "prior_style"] == 0.1
    assert base.loc[2, "prior_style"] == 0.2
    assert base.loc[3, "prior_style"] == 0.2  # missing race carries last observed value
    changed = style.copy()
    changed.loc[changed["round"] >= 2, "style"] = 99
    got = prior_style(X, changed)
    assert base.loc[1, "prior_style"] == got.loc[1, "prior_style"]
    changed.loc[changed["round"] == 1, "style"] = 50
    got = prior_style(X, changed)
    assert base.loc[1, "prior_style"] != got.loc[1, "prior_style"]


def test_cached_stints_ignore_target_and_later():
    X = table(); X = X[X.season <= 2025].copy()
    source = pd.read_csv(PART)
    base = prior_style(X, source)
    for year, round_ in ((2023, 7), (2025, 20)):
        changed = source.copy()
        late = (changed.season > year) | ((changed.season == year) & (changed["round"] >= round_))
        changed.loc[late, "style"] = 99
        got = prior_style(X, changed)
        target = lambda d: d[(d.season == year) & (d["round"] == round_)].sort_values("code").prior_style.reset_index(drop=True)
        pd.testing.assert_series_equal(target(base), target(got), check_exact=True)


if __name__ == "__main__":
    test_style_history_uses_only_past()
    test_cached_stints_ignore_target_and_later()
    print("tyre style leak test passed")
