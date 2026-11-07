"""E162: FP best-lap gap as one post-qualifying feature in the frozen compact logit."""
import glob, sys
import numpy as np, pandas as pd
import f1.walkforward as W
from experiments.common import table
from run import COMPACT, baselines


def main():
    X = table(); X = X[X.season <= 2025].copy()
    fp = pd.concat([pd.read_csv(f) for f in glob.glob("data/parts/fp_pace_*.csv")]).drop_duplicates(["season", "round", "session", "code"])
    a = fp.groupby(["season", "round", "code"]).apply(lambda g: np.average(g.fp_best, weights=g.w), include_groups=False).rename("fp_best").reset_index()
    X = X.merge(a, on=["season", "round", "code"], how="left")
    cover = X.groupby("season").fp_best.apply(lambda s: s.notna().mean()).round(2); print("fp coverage by season:", cover.to_dict())
    seasons = [s for s, c in cover.items() if c > 0.8 and s >= 2020]; print("evaluated seasons:", seasons)
    X["fp_best"] = X.fp_best.fillna(X.groupby(["season", "round"]).fp_best.transform("median")).fillna(X.fp_best.median())
    baselines()
    W.MODELS["logit_fp"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, COMPACT + ["fp_best"], C=0.3, hl=100)
    W.MODELS["fp_only"] = lambda tr, te, F, **kw: te.fp_best.values
    R, P = W.walk_forward(X, seasons, ["grid", "compact_logit", "logit_fp", "fp_only"], F=COMPACT, ens={"ens_lfp": ["compact_logit", "fp_only"]})
    print(R.groupby(["model", "season"]).podium.mean().unstack().round(3).to_string())
    print("pooled:", R.groupby("model").podium.mean().round(4).to_dict())
    z = R.pivot(index=["season", "round"], columns="model", values="podium")
    for c in ("logit_fp", "ens_lfp"):
        for ref in ("compact_logit", "grid"):
            d, ci, _ = W.paired_boot(z[c], z[ref]); print(f"{c} - {ref}: {d:+.4f} [{ci[0]:+.3f},{ci[1]:+.3f}] n={len(z)}")
        late = z[z.index.get_level_values(0) >= 2023]; d, ci, _ = W.paired_boot(late[c], late["compact_logit"]); print(f"  2023+ {c} - compact_logit: {d:+.4f} [{ci[0]:+.3f},{ci[1]:+.3f}] n={len(late)}")


if __name__ == "__main__":
    main()
