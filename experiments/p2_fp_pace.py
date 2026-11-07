"""E161: corrected practice long-run pace per driver-weekend -> data/parts/fp_pace_<years>.csv, then measurement gate vs qualifying."""
import sys, warnings
import numpy as np, pandas as pd, fastf1
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from f1.data import DATA
warnings.filterwarnings("ignore"); fastf1.Cache.enable_cache(str(DATA / "fastf1_cache")); fastf1.set_log_level("ERROR")


def session_pace(laps):
    l = laps[laps.LapTime.notna() & laps.PitInTime.isna() & laps.PitOutTime.isna() & laps.TrackStatus.astype(str).eq("1")
             & laps.IsAccurate.fillna(False) & laps.TyreLife.notna() & laps.Compound.notna()].copy()
    l["t"] = l.LapTime.dt.total_seconds(); l = l[l.t > 0]
    # consecutive kept laps within a stint: drop first lap, require >=4 kept laps
    l = l.sort_values(["Driver", "Stint", "LapNumber"]); l["i"] = l.groupby(["Driver", "Stint"]).cumcount()
    n = l.groupby(["Driver", "Stint"]).t.transform("size"); l = l[(n >= 4) & (l.i >= 1)]
    if l.Driver.nunique() < 8: return None
    drv = pd.get_dummies(l.Driver, dtype=float); comp = pd.get_dummies(l.Compound, prefix="c", dtype=float)
    age = comp.mul((l.TyreLife - l.groupby("Compound").TyreLife.transform("median")) / 10, axis=0).add_prefix("age_")
    design = pd.concat([drv, comp, age, (l.i / 10).rename("fuel"), (l.LapNumber / 30).rename("evo")], axis=1)
    y = np.log(l.t) - np.log(l.t).median()
    fit = Ridge(alpha=10, fit_intercept=True).fit(design, y)
    coef = pd.Series(fit.coef_[: drv.shape[1]], index=drv.columns)
    med = l.groupby("Driver").t.median(); med = (med / med.min() - 1) * 100
    best = laps[laps.LapTime.notna()].groupby("Driver").LapTime.min().dt.total_seconds(); best = (best / best.min() - 1) * 100
    return pd.DataFrame({"fp_corr": (coef - coef.min()) * 100, "fp_med": med, "fp_best": best, "n_lr": l.groupby("Driver").size()})


def extract(years):
    rows = []
    for y in years:
        sched = fastf1.get_event_schedule(y, include_testing=False)
        for _, ev in sched.iterrows():
            sprint = "sprint" in str(ev.EventFormat).lower()
            for ses, w in (("FP1", 1 if sprint else 0), ("FP2", 2), ("FP3", 1)):
                if w == 0: continue
                try:
                    s = ev.get_session(ses); s.load(laps=True, telemetry=False, weather=False, messages=False); laps = s.laps
                except Exception: continue
                p = session_pace(laps)
                if p is None: continue
                rows.append(p.reset_index(names="code").assign(season=y, round=int(ev.RoundNumber), session=ses, w=w)); print(y, ev.RoundNumber, ses, len(p), flush=True)
    out = pd.concat(rows); out.to_csv(DATA / f"parts/fp_pace_{years[0]}_{years[-1]}.csv", index=False); return out


def gate(fp):
    X = pd.read_pickle(DATA / "cache_X.pkl"); res = X[["season", "round", "code", "q_gap"]]; pace = pd.read_csv(DATA / "racepace_ff1.csv")
    agg = fp.assign(**{c: fp[c] * fp.w for c in ("fp_corr", "fp_med", "fp_best")}).groupby(["season", "round", "code"]).agg(
        fp_corr=("fp_corr", "sum"), fp_med=("fp_med", "sum"), fp_best=("fp_best", "sum"), w=("w", "sum")).reset_index()
    for c in ("fp_corr", "fp_med", "fp_best"): agg[c] /= agg.w
    d = res.merge(pace, on=["season", "round", "code"]).merge(agg, on=["season", "round", "code"], how="left")
    rows = []
    for (s, r), g in d.dropna(subset=["pace_gap"]).groupby(["season", "round"]):
        if g.fp_corr.notna().sum() < 10: continue
        rows.append(dict(season=s, round=r, n=int(g.fp_corr.notna().sum()), **{c: spearmanr(g[c], g.pace_gap, nan_policy="omit")[0] for c in ("q_gap", "fp_corr", "fp_med", "fp_best")}))
    R = pd.DataFrame(rows); print(R.groupby("season")[["q_gap", "fp_corr", "fp_med", "fp_best"]].mean().round(3).to_string()); print("races", R.groupby("season").size().to_dict())
    return R


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or [2020, 2023, 2025]
    gate(extract(years))
