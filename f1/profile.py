"""Pre-weekend profile features: car profile x circuit demands, driver style, strategy/attrition character of the circuit.
Everything for target race k is computed from weekends STRICTLY BEFORE k (telemetry/sessions/strategy of past weekends only) plus static
circuit traits taken from PAST visits to the same circuit. No data of the target weekend is touched (tests/test_preweekend.py)."""
import numpy as np
import pandas as pd

from f1.data import DATA

TEL = ["apex_slow", "apex_med", "apex_fast", "brake_slow", "brake_med", "brake_fast", "pick_slow", "pick_med", "pick_fast",
       "gain_slow", "gain_med", "gain_fast", "exitv_slow", "exitv_med", "exitv_fast", "acc_gap", "vmax_gap", "ft", "brk"]
SES = ["lr", "s1", "s2", "s3", "st", "q", "deg", "pit", "exec"]  # driver-weekend session metrics (gaps vs best)
CIRC = ["c_len", "c_slow", "c_med", "c_fast", "c_ft", "c_brk", "c_vmax", "c_vmean", "c_longest", "c_lowspeed", "c_ncor"]
STRAT = ["stops_med", "stops_2plus", "stint_len", "soft", "hard", "wet", "sc_frac", "vsc_frac"]
PU = {  # season-range -> {team: power-unit supplier}; hand-entered from public knowledge (assumption, not verified against a data source)
    (2018, 2020): dict(mercedes="mer", williams="mer", aston_martin="mer", ferrari="fer", haas="fer", sauber="fer", red_bull="ren", mclaren="ren", alpine="ren", rb="hon"),
    (2021, 2021): dict(mercedes="mer", williams="mer", aston_martin="mer", mclaren="mer", ferrari="fer", haas="fer", sauber="fer", red_bull="hon", rb="hon", alpine="ren"),
    (2022, 2024): dict(mercedes="mer", williams="mer", aston_martin="mer", mclaren="mer", ferrari="fer", haas="fer", sauber="fer", red_bull="hon", rb="hon", alpine="ren"),
    (2025, 2025): dict(mercedes="mer", williams="mer", aston_martin="mer", mclaren="mer", alpine="mer", ferrari="fer", haas="fer", sauber="fer", red_bull="hon", rb="hon"),
    (2026, 2030): dict(mercedes="mer", williams="mer", mclaren="mer", alpine="mer", ferrari="fer", haas="fer", cadillac="fer", red_bull="rbf", rb="rbf", aston_martin="hon", sauber="aud"),
}
PU_FAIL = ["Engine", "Power Unit", "Power loss", "Electrical", "Turbo", "Exhaust", "Overheating", "Fuel pressure", "Oil leak", "Water pressure", "Energy recovery", "ERS", "MGU-K"]
COLL = ["Collision", "Accident", "Collision damage", "Spun off"]


def _cat(name):
    fs = sorted((DATA / "parts").glob(f"{name}_*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def load_aux():
    tel, circ, strat, ses, pits = _cat("tel"), _cat("circ"), _cat("strat"), _cat("sessions"), _cat("pits")
    pace = pd.read_csv(DATA / "racepace_ff1.csv")
    for d in (tel, circ, strat, ses, pits):
        if len(d):
            d.drop_duplicates(subset=[c for c in ("season", "round", "code", "session") if c in d], inplace=True)
    return tel, circ, strat, ses, pits, pace


def _driver_weekend(df, tel, ses, pits, pace):
    """one row per (season, round, code): telemetry + session metrics + incident flag, plus team-mean and driver-minus-teammate versions."""
    base = df[["season", "round", "code", "team", "status", "order"]].copy()
    base["coll"] = base.status.isin(COLL).astype(float)
    base["pu"] = [next((m.get(t) for (a, b), m in PU.items() if a <= y <= b), None) for y, t in zip(base.season, base.team)]
    base["pufail"] = base.status.isin(PU_FAIL).astype(float)  # engine-related retirement
    parts = [base]
    if len(tel):
        parts.append(tel[["season", "round", "code"] + [c for c in TEL if c in tel]])
    w = base
    for p in parts[1:]:
        w = w.merge(p, on=["season", "round", "code"], how="left")
    if len(ses):
        fp = ses[ses.session.isin(["FP1", "FP2", "FP3"])].groupby(["season", "round", "code"])["lr_gap"].mean().rename("lr")
        q = ses[ses.session == "Q"].set_index(["season", "round", "code"])[["s1_gap", "s2_gap", "s3_gap", "st_gap", "best_gap"]]
        q.columns = ["s1", "s2", "s3", "st", "q"]
        r = ses[ses.session == "R"].set_index(["season", "round", "code"])[["deg"]]
        w = w.merge(pd.concat([fp, q, r], axis=1).reset_index(), on=["season", "round", "code"], how="left")
    if len(pits):
        w = w.merge(pits[["season", "round", "code", "pit_gap"]].rename(columns={"pit_gap": "pit"}), on=["season", "round", "code"], how="left")
    # strategy/race-craft execution: finishing rank vs race-pace rank among classified cars (positive = beat what pace alone predicts)
    fin = w[w.status.isin(["Finished", "Lapped"]) | w.status.str.startswith("+")].merge(pace, on=["season", "round", "code"], how="inner")
    fin["exec"] = fin.groupby(["season", "round"]).pace_gap.rank() - fin.groupby(["season", "round"]).order.rank()
    w = w.merge(fin[["season", "round", "code", "exec"]], on=["season", "round", "code"], how="left")
    m = [c for c in TEL + SES if c in w]
    key = ["season", "round", "team"]
    for c in m:  # team = mean of its drivers; driver style = own value minus the TEAMMATE-average in the same car & weekend
        tm = w.groupby(key)[c].transform("mean"); n = w.groupby(key)[c].transform("count")
        w["t_" + c] = tm
        w["s_" + c] = np.where(n >= 2, w[c] - tm, np.nan)
    return w, m


def _circuit_table(df, circ, strat):
    """per-race circuit character: layout traits (telemetry), strategy, attrition. Keyed by (season, round); aggregated over PAST visits later."""
    r = df.drop_duplicates(["season", "round"])[["season", "round", "circuit"]].copy()
    d = df.assign(dnfx=(~(df.status.isin(["Finished", "Lapped"]) | df.status.str.startswith("+")) & ~df.status.isin(["Did not start", "Withdrew"])).astype(float),
                  collx=df.status.isin(COLL).astype(float),
                  chaos=((df.order <= 3) & (df.grid > 3)).astype(float) * (df.grid > 0) + ((df.order <= 3) & (df.grid == 0)))
    att = d.groupby(["season", "round"]).agg(c_dnf=("dnfx", "mean"), c_coll=("collx", "mean"), c_chaos=("chaos", "sum")).reset_index()
    att["c_chaos"] /= 3
    r = r.merge(att, on=["season", "round"], how="left")
    if len(circ):
        r = r.merge(circ[["season", "round"] + [c for c in CIRC if c in circ]], on=["season", "round"], how="left")
    if len(strat):
        r = r.merge(strat[["season", "round"] + [c for c in STRAT if c in strat]], on=["season", "round"], how="left")
    return r


def add_profile(X, df, aux=None, hl=6, era_mult=0.5):
    """X: output of build_features (pre-weekend features exist). Returns X + profile/circuit/fit features."""
    tel, circ, strat, ses, pits, pace = aux if aux is not None else load_aux()
    w, M = _driver_weekend(df, tel, ses, pits, pace)
    k_of = X.drop_duplicates(["season", "round"])[["season", "round", "k", "era"]]
    w = w.merge(k_of, on=["season", "round"]); ct = _circuit_table(df, circ, strat).merge(k_of, on=["season", "round"])
    ctrait = [c for c in ct if c.startswith("c_") or c in STRAT]
    out = []
    for k, cur in X.groupby("k", sort=True):
        f = cur[["season", "round", "driver", "team", "code"]].copy()
        f["pu"] = [next((m.get(t) for (a, b), m in PU.items() if a <= y <= b), None) for y, t in zip(cur.season, cur.team)]
        pw, pc = w[w.k < k], ct[ct.k < k]
        if len(pw):
            age = k - pw.k; wt = 0.5 ** (age / hl) * np.where(pw.era < cur.era.iloc[0], era_mult, 1.0)
            for pre, key in (("tp_", "team"), ("ds_", "code")):
                cols = [("t_" if pre == "tp_" else "s_") + c for c in M] + (["coll"] if pre == "ds_" else [])
                v = pw[cols]; num = v.mul(wt, axis=0).where(v.notna()); den = v.notna().mul(wt, axis=0)
                agg = num.groupby(pw[key]).sum() / den.groupby(pw[key]).sum().replace(0, np.nan)
                agg.columns = [pre + c.split("_", 1)[1] if c != "coll" else "d_coll" for c in cols]
                f = f.join(agg, on="team" if pre == "tp_" else "code")
        if len(pw):
            for c in [c for c in ("t_acc_gap", "t_vmax_gap", "pufail") if c in pw]:  # PU pooled across works + customer cars (same supplier in that season)
                v = pw[c]; num = (v * wt).where(v.notna()).groupby([pw.pu]).sum(); den = (v.notna() * wt).groupby([pw.pu]).sum().replace(0, np.nan)
                f["pu_" + c.replace("t_", "")] = f.pu.map(num / den)
        # circuit: static traits from the most recent PAST visits to this track (layout/strategy), attrition averaged over past visits
        circuit = df[(df.season == cur.season.iloc[0]) & (df["round"] == cur["round"].iloc[0])].circuit.iloc[0]
        pv = pc[pc.circuit == circuit].sort_values("k")
        if len(pv):
            last = pv.tail(2)[ctrait].mean()
            for c in ctrait:
                f[c] = last[c]
            for c in ("c_dnf", "c_coll", "c_chaos"):
                f[c] = pv[c].mean() if c in pv else np.nan
        out.append(f)
    P = pd.concat(out, ignore_index=True)
    X = X.merge(P.drop(columns=["team", "code", "pu"]), on=["season", "round", "driver"], how="left")
    tot = X[["c_slow", "c_med", "c_fast"]].sum(axis=1).replace(0, np.nan)
    for k_ in ("slow", "med", "fast"):
        X["f_" + k_] = X["c_" + k_] / tot
    X["f_straight"] = X.get("c_ft")
    for who, pre in (("car", "tp_"), ("style", "ds_")):  # profile x circuit demands ("fit")
        for m in ("apex", "brake", "pick", "gain", "exitv"):
            cs = [pre + f"{m}_{k_}" for k_ in ("slow", "med", "fast")]
            if all(c in X for c in cs):
                X[f"fit_{who}_{m}"] = sum(X[c] * X["f_" + k_] for c, k_ in zip(cs, ("slow", "med", "fast")))
        if pre + "acc_gap" in X:
            X[f"fit_{who}_acc"] = X[pre + "acc_gap"] * X["c_ft"]; X[f"fit_{who}_vmax"] = X[pre + "vmax_gap"] * X["c_vmax"] / 330
        if pre + "deg" in X and "stops_med" in X:
            X[f"fit_{who}_deg"] = X[pre + "deg"] * X["stops_med"]
    return X


PROFILE_COLS = lambda X: [c for c in X if c.startswith(("tp_", "ds_", "fit_", "f_", "pu_")) or c in
                          ("c_len", "c_slow", "c_med", "c_fast", "c_ft", "c_brk", "c_vmax", "c_vmean", "c_longest", "c_lowspeed", "c_ncor",
                           "c_dnf", "c_coll", "c_chaos", "d_coll") + tuple(STRAT)]
