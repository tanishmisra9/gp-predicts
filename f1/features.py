"""Leak-free feature builder. Row k's features use ONLY (a) results of races strictly before k and
(b) pre-race info of race k itself (grid, qualifying, sprint, practice laps).
tests/test_no_leak.py proves it by scrambling all race-k-and-later outcomes and checking features don't move."""
import numpy as np
import pandas as pd

from f1.data import DATA

OUTCOME = ["order", "finish", "status", "points", "laps"]  # known only after the race
ERA = lambda s: 0 if s <= 2021 else (1 if s <= 2025 else 2)  # regulation eras: 2017-21, 2022-25, 2026-


def _prep(df, laps=None):
    df = df.copy().sort_values(["season", "round"]).reset_index(drop=True)
    df["k"] = df.groupby(["season", "round"], sort=False).ngroup()
    df["era"] = df.season.map(ERA)
    df["gridn"] = df.grid.where(df.grid > 0, 21)  # pit-lane start -> back of the field
    for c in ("q1", "q2", "q3"):
        df[c + "_gap"] = (df[c] / df.groupby("k")[c].transform("min") - 1) * 100
    df["q_gap"] = df.q3_gap.fillna(df.q2_gap).fillna(df.q1_gap)  # deepest stage reached
    df["q_gap_best"] = df.groupby("k").q_gap.transform("min")
    if laps is not None:
        if "session" in laps:  # OpenF1 table: practice/quali gaps per session + race pace
            vals = [c for c in ("best_gap", "top5_gap", "lr_gap") if c in laps]
            lp = laps.pivot_table(index=["season", "round", "code"], columns="session", values=vals)
            lp.columns = [f"{a}_{b}" for a, b in lp.columns]
            df = df.merge(lp.reset_index(), on=["season", "round", "code"], how="left")
            for kind in ("best", "top5", "lr"):  # sprint weekends only have FP1 -> average whichever FP sessions exist
                cols = [f"{kind}_gap_{x}" for x in ("FP1", "FP2", "FP3") if f"{kind}_gap_{x}" in df]
                df[f"fp_{kind}"] = df[cols].mean(axis=1)
            rp = laps[laps.session == "R"][["season", "round", "code", "pace_gap"]]
        else:  # FastF1 race-pace table: season, round, code, pace_gap
            rp = laps[["season", "round", "code", "pace_gap"]]
        df = df.merge(rp, on=["season", "round", "code"], how="left")
    # post-race (past-only) quantities -------------------------------------------------
    df["dnf"] = (~(df.status.isin(["Finished", "Lapped"]) | df.status.str.startswith("+"))).astype(float)
    df["gain"] = df.gridn - df.order
    df["pts_n"] = df.points / 25
    df["win"] = (df.order == 1).astype(int)
    df["podium"] = (df.order <= 3).astype(int)
    return df


PAST_COLS = ["order", "q_pos", "q_gap", "pts_n", "dnf", "gain"]


def _wmean(frame, keys, cols, w):
    f = frame[cols].mul(w, axis=0).where(frame[cols].notna())
    ws = frame[cols].notna().mul(w, axis=0)
    return f.groupby([frame[k] for k in keys]).sum() / ws.groupby([frame[k] for k in keys]).sum().replace(0, np.nan)


def build_features(df, laps=None, hl=(3, 10), era_mult=(0.3, 0.1)):
    """Return one feature row per driver-race. era_mult=(driver, team): weight multiplier on rows from an older era."""
    df = _prep(df, laps)
    pcols = PAST_COLS + (["pace_gap"] if "pace_gap" in df else [])
    out = []
    for k, cur in df.groupby("k", sort=True):
        past = df[df.k < k]
        f = cur[["season", "round", "k", "driver", "team", "code", "era", "gridn", "q_pos", "q_gap", "q_gap_best",
                 "s_grid", "s_order"] + [c for c in df if c.startswith(("best_gap_", "top5_gap_", "lr_gap_", "fp_"))]
                + ["order", "podium", "win"]].copy()
        f["grid_pen"] = f.gridn - f.q_pos.fillna(f.gridn)
        f["season_round"] = cur["round"].values
        f["era_round"] = (f.season - {0: 2017, 1: 2022, 2: 2026}[ERA(f.season.iloc[0])]) * 22 + f["round"]
        f["field"] = len(cur)
        # weekend-relative: team's best car this weekend, and gap to teammate
        g = f.groupby("team")
        f["team_qgap"] = g.q_gap.transform("min")
        f["tm_qgap"] = f.q_gap - (g.q_gap.transform("sum") - f.q_gap)  # vs teammate (NaN if no mate)
        f["team_grid"] = g.gridn.transform("mean")
        for c in [c for c in f if c.startswith(("best_gap_", "top5_gap_", "lr_gap_", "fp_"))]:
            f["team_" + c] = g[c].transform("min")
        f["grid_rank_in_team"] = g.gridn.rank()
        if len(past):
            age = k - past.k
            for h in hl:
                for who, key, em in (("d", "driver", era_mult[0]), ("t", "team", era_mult[1])):
                    w = 0.5 ** (age / h) * np.where(past.era < cur.era.iloc[0], em, 1.0)
                    m = _wmean(past, [key], pcols, pd.Series(w, index=past.index))
                    m.columns = [f"{who}{h}_{c}" for c in pcols]
                    f = f.join(m, on=key)
            # season-to-date standings share (this season only)
            ps = past[past.season == cur.season.iloc[0]]
            if len(ps):
                dp = ps.groupby("driver").pts_n.sum(); tp = ps.groupby("team").pts_n.sum()
                f["d_season_pts"] = f.driver.map(dp) / max(dp.max(), 1e-9); f["t_season_pts"] = f.team.map(tp) / max(tp.max(), 1e-9)
                f["season_races"] = ps.k.nunique()
            # circuit character: how strongly grid predicted finish here historically (past races only)
            circ = cur.circuit.iloc[0]
            pc = past[past.circuit == circ]
            # circuit-specific form: how each team/driver ran HERE in past seasons relative to their own season average
            pcs = pc[pc.season < cur.season.iloc[0]]
            if len(pcs):
                for who in ("team", "driver"):
                    here = pcs.groupby(["season", who]).order.mean()
                    allr = past[past.season.isin(pcs.season.unique())].groupby(["season", who]).order.mean()
                    res = (here - allr).dropna().reset_index(name="r").sort_values("season")
                    res = res.groupby(who).r.apply(lambda x: x.tail(3).mean())
                    f[("t" if who == "team" else "d") + "_circ_res"] = f[who].map(res)
            if pc.k.nunique():
                cc = pc.groupby("k").apply(lambda x: x.gridn.corr(x.order, method="spearman"), include_groups=False)
                f["circ_grid_corr"] = cc.tail(4).mean()
                f["circ_pole_win"] = pc[pc.gridn == 1].win.mean()
        out.append(f)
    out = pd.concat(out, ignore_index=True).sort_values(["k", "driver"]).reset_index(drop=True)
    # previous race's finishing position for the same driver (strictly the race before; NaN if none)
    out["prev_order"] = out.sort_values("k").groupby("driver")["order"].shift(1)
    out["log_grid"] = np.log(out.gridn)
    out["grid_x_corr"] = out.log_grid * out.circ_grid_corr.fillna(0.7) if "circ_grid_corr" in out else np.nan
    return out


if __name__ == "__main__":
    from f1.data import load_results
    X = build_features(load_results())
    print(X.shape); print(X.tail(3).T.to_string())


# Features usable BEFORE the weekend starts: only decayed history of earlier races + static context.
# (everything keyed on the target weekend's grid / quali / sprint / practice is excluded by construction)
def pre_weekend_cols(X):
    hist = [c for c in X if c.split("_", 1)[0] in ("d3", "d10", "t3", "t10")]
    extra = ["d_season_pts", "t_season_pts", "season_races", "circ_grid_corr", "circ_pole_win", "era_round", "field",
             "prev_order", "t_circ_res", "d_circ_res"]
    return hist + [c for c in extra if c in X]
