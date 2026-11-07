"""E160: post-qualifying ensembles of grid, standings and the frozen compact logit."""
import numpy as np
import pandas as pd
import f1.walkforward as W
from experiments.common import table
from run import COMPACT, baselines


def main():
    X = table(); X = X[X.season <= 2025]
    baselines()
    R, P = W.walk_forward(X, range(2020, 2026), ["grid", "standings", "compact_logit"], F=COMPACT,
                          ens={"ens_gl": ["grid", "compact_logit"], "ens_gsl": ["grid", "standings", "compact_logit"],
                               "ens_gs": ["grid", "standings"]})
    R["sp"] = np.where(R.season <= 2022, "inner", "outer")
    print(R.groupby(["model", "sp"]).podium.mean().unstack().round(4).sort_values("outer", ascending=False).to_string())
    print("\nby season:\n", R.groupby(["model", "season"]).podium.mean().unstack().round(3).to_string())
    for sp in ("inner", "outer"):
        z = R[R.sp == sp].pivot(index=["season", "round"], columns="model", values="podium")
        for c in ("ens_gl", "ens_gsl", "ens_gs", "compact_logit"):
            for ref in ("grid", "compact_logit"):
                if c != ref:
                    d, ci, _ = W.paired_boot(z[c], z[ref]); print(f"{sp} {c} - {ref}: {d:+.4f} [{ci[0]:+.3f},{ci[1]:+.3f}] n={len(z)}")


if __name__ == "__main__":
    main()
