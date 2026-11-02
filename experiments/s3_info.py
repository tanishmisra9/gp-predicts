import sys; sys.path.insert(0, ".")
from experiments.common import *
X = table(); BASE = pre_weekend_cols(X); DEV = [2020, 2021, 2022, 2023, 2024, 2025]
res = load_results()
# ---- E69 championship context (all from points BEFORE the race)
X["nrounds"] = X.groupby("season")["round"].transform("max"); X["rounds_left"] = X.nrounds - X["round"]
X["gap_leader"] = 1 - X.d_season_pts; X["tgap_leader"] = 1 - X.t_season_pts
second = X.groupby("k").d_season_pts.transform(lambda s: s.nlargest(2).iloc[-1] if s.notna().sum() > 1 else np.nan); X["lead_margin"] = 1 - second
X["title_live"] = X.gap_leader * 25 / X.rounds_left.clip(lower=1).astype(float)                 # leader gap per race left (title still reachable if small)
X["lateseason"] = X["round"] / X.nrounds
# ---- E70 rookie / experience, team-prior imputation
X["d_exp"] = X.sort_values("k").groupby("driver").cumcount(); X["rookie"] = (X.d_exp < 10).astype(float)
for c in ("d3_order", "d10_order", "d3_q_pos", "d10_q_pos", "d10_pace_gap"):
    X[c + "_imp"] = X[c].fillna(X[c.replace("d", "t", 1)]) if c.replace("d", "t", 1) in X else X[c]
# ---- E71 calendar / rest
dt = res.drop_duplicates(["season", "round"]).assign(date=lambda d: pd.to_datetime(d.date)).sort_values("date"); dt["gap_days"] = dt.date.diff().dt.days
dt["b2b"] = (dt.gap_days <= 8).astype(float); X = X.merge(dt[["season", "round", "gap_days", "b2b"]], on=["season", "round"], how="left")
# ---- E72 within-team strength share (decayed: driver's share of team points)
tm = X.groupby(["k", "team"]).d10_pts_n.transform("sum"); X["d_share"] = X.d10_pts_n / tm.replace(0, np.nan)
X["share_x_form"] = X.d_share.fillna(.5) * X.t10_order.fillna(11)
# ---- E73 stage-of-season interactions: shrink recent form toward long-run early in a season
sr = X.season_races.fillna(0); X["w_new"] = sr / (sr + 4)
X["blend_order"] = X.w_new * X.d3_order.fillna(11) + (1 - X.w_new) * X.d10_order.fillna(11); X["blend_pace"] = X.w_new * X.t3_pace_gap.fillna(1) + (1 - X.w_new) * X.t10_pace_gap.fillna(1)
blocks = {"E69_championship_context": ["gap_leader", "tgap_leader", "lead_margin", "title_live", "rounds_left", "lateseason"],
          "E70_rookie_experience_teamprior": ["d_exp", "rookie", "d3_order_imp", "d10_order_imp", "d3_q_pos_imp", "d10_q_pos_imp", "d10_pace_gap_imp"],
          "E71_calendar_rest": ["gap_days", "b2b"], "E72_within_team_share": ["d_share", "share_x_form"], "E73_season_stage_blend": ["w_new", "blend_order", "blend_pace"]}
W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
W.MODELS["logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, BASE, C=0.03, hl=100)
for nm, ex in blocks.items():
    W.MODELS[nm] = (lambda ex: lambda tr, te, F, **kw: W.m_logit_pod(tr, te, BASE + ex, C=0.03, hl=100))(ex)
    W.MODELS[nm + "_ET"] = (lambda ex: lambda tr, te, F, **kw: reg(tr, te, BASE + ex, "y_pace", "et"))(ex)
W.MODELS["E73b_all_blocks"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, BASE + sum(blocks.values(), []), C=0.03, hl=100)
W.MODELS["ET_base"] = lambda tr, te, F, **kw: reg(tr, te, BASE, "y_pace", "et")
ms = ["standings", "logit", "ET_base"] + [m for m in W.MODELS if m.startswith("E7") or m.startswith("E69")]
R, P = W.walk_forward(X, DEV, ms, ens={"ens": ["standings", "logit"]}, tie=None); report(R, "ens")
