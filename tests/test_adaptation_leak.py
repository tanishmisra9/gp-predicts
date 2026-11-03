"""Changing a transfer's later pace cannot change its pre-window forecast."""

import pandas as pd

from experiments.b6_adaptation_gate import forecast, transfer_windows
from f1.data import DATA, load_results


def test_transfer_cutoff():
    results = load_results().query("season <= 2025")
    pace = pd.read_csv(DATA / "racepace_ff1.csv")
    original = forecast(transfer_windows(results, pace)).set_index(["code", "season"])
    target = original.loc["HAM", 2025]
    later = pace.copy()
    mask = later.code.eq("HAM") & later.season.eq(2025) & later["round"].ge(6)
    later.loc[mask, "pace_gap"] += 100
    changed = forecast(transfer_windows(results, later)).set_index(["code", "season"]).loc["HAM", 2025]
    assert changed.early == target.early and changed.adapted == target.adapted
    assert changed.late != target.late
    earlier = pace.copy()
    mask = earlier.code.eq("HAM") & earlier.season.eq(2025) & earlier["round"].le(5)
    earlier.loc[mask, "pace_gap"] += 1
    changed = forecast(transfer_windows(results, earlier)).set_index(["code", "season"]).loc["HAM", 2025]
    assert changed.adapted != target.adapted


if __name__ == "__main__":
    test_transfer_cutoff()
    print("adaptation leak test passed")
