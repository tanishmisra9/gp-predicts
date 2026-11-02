import sys, glob; sys.path.insert(0, ".")
from experiments.common import *
from scipy.stats import spearmanr
X = table(); res = load_results()
pv = pd.concat([pd.read_csv(f) for f in glob.glob("data/parts/pacev_*.csv")]).drop_duplicates(["season", "round", "code"])
D = X[["season", "round", "k", "code", "team", "order", "era"]].merge(pv, on=["season", "round", "code"], how="left").merge(res[["season", "round", "code", "status"]], on=["season", "round", "code"], how="left")
D["fin"] = D.status.isin(["Finished", "Lapped"]) | D.status.str.startswith("+")
V = [c for c in pv if c.endswith("_gap")]
print("variants:", V, "| rows with values:", {c: int(D[c].notna().sum()) for c in V})
def past(col, key, hl):
    out = []
    for k, era in D.drop_duplicates("k")[["k", "era"]].values:
        p = D[D.k < k].dropna(subset=[col])
        if p.empty: continue
        w = 0.5 ** ((k - p.k) / hl) * np.where(p.era < era, 0.5, 1.0); g = (p[col] * w).groupby(p[key]).sum() / w.groupby(p[key]).sum(); out.append(pd.DataFrame({"k": k, key: g.index, "v": g.values}))
    return pd.concat(out)
rows = []
for c in V:
    for hl in (4, 10):
        f = past(c, "code", hl).rename(columns={"v": "f"}); m = D.merge(f, on=["k", "code"]).dropna(subset=["f"])
        m = m[m.season.between(2020, 2025)]
        r_fin = m[m.fin].groupby("k").apply(lambda g: spearmanr(g.f, g.order)[0] if len(g) > 6 else np.nan, include_groups=False)           # forecast vs actual FINISHING order (classified cars)
        r_same = m.dropna(subset=[c]).groupby("k").apply(lambda g: spearmanr(g.f, g[c])[0] if len(g) > 6 else np.nan, include_groups=False)   # forecast vs the same measure next race
        rows.append((c, hl, r_fin.mean(), r_same.mean()))
R = pd.DataFrame(rows, columns=["pace measure", "half-life", "corr w/ next race FINISHING order", "corr w/ next race same measure"]); print(R.round(4).to_string(index=False))
