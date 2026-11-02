import sys; sys.path.insert(0, ".")
from experiments.common import *
from f1.profile import load_aux, _driver_weekend, TEL
X = table(); tel, circ, strat, ses, pits, pace = load_aux(); df = load_results()
w, M = _driver_weekend(df, tel, ses, pits, pace)
fp = ses[ses.session.isin(["FP1", "FP2", "FP3"])].groupby(["season", "round", "code"]).agg(fp_best=("best_gap", "mean"), fp_lr=("lr_gap", "mean")).reset_index()
cur = w[["season", "round", "code"] + [c for c in TEL if c in w] + ["s1", "s2", "s3", "st"]].copy()
cur.columns = ["season", "round", "code"] + ["cur_" + c for c in cur.columns[3:]]
X = X.merge(fp, on=["season", "round", "code"], how="left").merge(cur, on=["season", "round", "code"], how="left")
for c in ("fp_best", "fp_lr"): X[c + "_rank"] = X.groupby("k")[c].rank()
BASE = pre_weekend_cols(X); DEV = [2020, 2021, 2022, 2023, 2024, 2025]
S1 = ["fp_best", "fp_lr", "fp_best_rank", "fp_lr_rank"]
S2 = ["gridn", "log_grid", "q_pos", "q_gap", "grid_pen", "team_qgap", "tm_qgap", "team_grid", "circ_grid_corr", "grid_x_corr"]
S3 = [c for c in X if c.startswith("cur_")]
u = lambda l: list(dict.fromkeys(l))
stages = {"S0 pre-weekend (history only)": u(BASE), "S1 + practice pace": u(BASE + S1), "S2 + grid & qualifying": u(BASE + S1 + S2), "S3 + this weekend's quali corner/engine telemetry": u(BASE + S1 + S2 + S3)}
for nm, F in stages.items():
    W.MODELS[f"lg|{nm}"] = lambda tr, te, F_, F=F, **kw: W.m_logit_pod(tr, te, F, C=0.05, hl=100)
    W.MODELS[f"gb|{nm}"] = lambda tr, te, F_, F=F, **kw: W.m_lgb_pod(tr, te, F, hl=100)
W.MODELS["grid_only"] = lambda tr, te, F, **kw: te.gridn.values
R, P = W.walk_forward(X, DEV, ["grid_only"] + [m for m in W.MODELS if "|" in m], tie="gridn")   # grid IS legitimately known from S2 on
s = W.summarize(R)[["winner", "podium", "ndcg3", "top10", "pairwise", "mae"]]
print(s.sort_index().to_string())
for st in stages:
    a = R[R.model == f"lg|{st}"].set_index(["season", "round"]).podium; b = R[R.model == "grid_only"].set_index(["season", "round"]).podium
    print(f"{st[:48]:48s} podium logit - grid: {W.paired_boot(a, b)[0]:+.3f}")
# per-season for the most informed stage
print(R.groupby(["model", "season"]).podium.mean().unstack().round(3).loc[[f"lg|{list(stages)[-1]}", f"gb|{list(stages)[-1]}", "grid_only"]].to_string())
