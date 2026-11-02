import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from f1.data import load_results, DATA
from f1.profile import load_aux
df = load_results(); pace = pd.read_csv(DATA / "racepace_ff1.csv"); tel, circ, strat, ses, pits, _ = load_aux()
d = df.merge(pace, on=["season", "round", "code"], how="left")
d["k"] = d.groupby(["season", "round"], sort=False).ngroup()
c = circ.merge(df.drop_duplicates(["season", "round"])[["season", "round", "circuit"]], on=["season", "round"])
c["f_slow"] = c.c_slow / (c.c_slow + c.c_med + c.c_fast); c["f_fast"] = c.c_fast / (c.c_slow + c.c_med + c.c_fast)
T = c.groupby("circuit")[["f_slow", "f_fast", "c_ft", "c_vmax", "c_lowspeed", "c_longest", "c_len"]].mean(); T = (T - T.mean()) / T.std()
tw = d.groupby(["k", "season", "round", "circuit", "team"]).pace_gap.mean().reset_index()   # team-weekend pace (lower=faster)
tw = tw.sort_values("k")
races = tw.drop_duplicates("k")[["k", "season", "circuit"]].reset_index(drop=True)
def forecast(tau, hl=4.0, same_season_only=False):
    """for each race, per-team kernel-weighted mean of past team pace: weight = recency * exp(-trait_dist^2/(2 tau^2))"""
    out = []
    for _, r in races.iterrows():
        past = tw[(tw.k < r.k) & tw.pace_gap.notna()]
        if same_season_only: past = past[past.season == r.season]
        else: past = past[past.season >= r.season - 1]
        if past.empty or r.circuit not in T.index: continue
        age = r.k - past.k; w = 0.5 ** (age / hl)
        if tau is not None:
            dist = past.circuit.map(lambda x: np.linalg.norm(T.loc[x] - T.loc[r.circuit]) if x in T.index else np.nan)
            w = w * np.exp(-(dist ** 2) / (2 * tau ** 2)); w = w.fillna(0)
        g = pd.DataFrame({"team": past.team, "w": w, "wp": w * past.pace_gap}).groupby("team").sum()
        f = (g.wp / g.w.replace(0, np.nan)).rename("fc").reset_index(); f["k"] = r.k; out.append(f)
    return pd.concat(out)
cur = tw.rename(columns={"pace_gap": "true"})
def score(f, lo=2023):
    m = cur.merge(f, on=["k", "team"]); m = m[m.season >= lo].dropna(subset=["true", "fc"])
    return m.groupby("k").apply(lambda g: spearmanr(g.fc, g.true)[0] if len(g) > 4 else np.nan, include_groups=False).mean()
print("TEAM-level pace-order forecast quality (Spearman vs true team race pace), 2023-26 races:")
print(f"  recency only (hl=4):                         {score(forecast(None)):.4f}")
for tau in (0.7, 1.0, 1.5, 2.5):
    print(f"  recency x track-similarity (tau={tau}):        {score(forecast(tau)):.4f}")
for hl in (2.0, 8.0):
    print(f"  recency only (hl={hl}):                        {score(forecast(None, hl=hl)):.4f}")
# --- how much pace-prediction accuracy does a given podium accuracy need? corrupt the TRUE driver race pace with noise ---
dd = d.dropna(subset=["pace_gap"]); rng = np.random.default_rng(0); rows = []
for sd in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2):
    ov, rho = [], []
    for (s, r), g in dd[dd.season.between(2023, 2026)].groupby(["season", "round"]):
        pred = g.pace_gap.values + rng.normal(0, sd, len(g)) * g.pace_gap.std()
        ov.append(len(set(g.driver.values[np.argsort(pred)[:3]]) & set(g.sort_values("order").driver.head(3))) / 3); rho.append(spearmanr(pred, g.pace_gap)[0])
    rows.append((sd, np.mean(rho), np.mean(ov)))
print("\nCorrupting TRUE race pace with noise -> (pace rank-correlation, podium accuracy):")
for sd, rho, ov in rows: print(f"  noise sd={sd:.1f}x  rho={rho:.3f}  podium={ov:.3f}")
