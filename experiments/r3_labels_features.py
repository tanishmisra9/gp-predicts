import sys, glob, itertools; sys.path.insert(0, ".")
from experiments.common import *
from sklearn.linear_model import LogisticRegression, Ridge
X = table(); BASE = pre_weekend_cols(X); DEV = [2020, 2021, 2022, 2023, 2024, 2025]
res = load_results()
X = X.merge(res[["season", "round", "code", "status", "points", "q1"]], on=["season", "round", "code"], how="left")  # s_order already in X (this weekend's sprint result; only ever used via PAST decayed means)
st = pd.concat([pd.read_csv(f) for f in glob.glob("data/parts/start_*.csv")]).drop_duplicates(["season", "round", "code"])
X = X.merge(st, on=["season", "round", "code"], how="left")
X["dnf_y"] = (~(X.status.isin(["Finished", "Lapped"]) | X.status.fillna("").str.startswith("+"))).astype(float)
SM = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
LG = lambda tr, te, F=None, **kw: W.m_logit_pod(tr, te, BASE, C=0.03, hl=100)
W.MODELS.update(standings=SM, logit=LG)
def pipe(C=0.03): return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, max_iter=500))
# ---------- generic past-only decayed signal
def decayed(S, col, key, hl, era_mult=0.5):
    """S: rows [k, era, key, col] of PAST outcomes -> for every target k, decayed mean per key using only k' < k"""
    out = []
    for k, era in X.drop_duplicates("k")[["k", "era"]].values:
        p = S[S.k < k].dropna(subset=[col]);
        if p.empty: continue
        w = 0.5 ** ((k - p.k) / hl) * np.where(p.era < era, era_mult, 1.0)
        g = (p[col] * w).groupby(p[key]).sum() / w.groupby(p[key]).sum(); out.append(pd.DataFrame({"k": k, key: g.index, "v": g.values}))
    return pd.concat(out)
def add(name, S, col, key, hl):
    global X
    d = decayed(S, col, key, hl).rename(columns={"v": name}); X = X.merge(d, on=["k", key], how="left")
X["q1gap"] = (X.q1 / X.groupby("k").q1.transform("min") - 1) * 100
X["lap1gain"] = X.gridn - X.pos_lap1                         # places gained on lap 1 (post-race info, used only from PAST races)
X["s_form"] = X.s_order
for hl in (4, 10):
    add(f"d{hl}_q1gap", X, "q1gap", "code", hl); add(f"t{hl}_q1gap", X, "q1gap", "team", hl)
add("d10_lap1gain", X, "lap1gain", "code", 10); add("d10_led", X, "led_share", "code", 10); add("t10_led", X, "led_share", "team", 10); add("t4_led", X, "led_share", "team", 4)
add("d10_sprint", X, "s_form", "code", 10)
# trend: team race-pace level recent (hl 3) minus longer (hl 10); same for results
X["t_trend_pace"] = X.t3_pace_gap - X.t10_pace_gap; X["d_trend_pace"] = X.d3_pace_gap - X.d10_pace_gap
X["t_trend_ord"] = X.t3_order - X.t10_order; X["d_trend_ord"] = X.d3_order - X.d10_order
# circuit continuity: did this team's driver podium at this circuit last visit?
res2 = res.copy(); res2["pod"] = (res2.order <= 3).astype(float); cc = X[["k", "season", "round", "code", "team", "circuit"] if "circuit" in X else ["k", "season", "round", "code", "team"]]
X = X.merge(res[["season", "round", "circuit"]].drop_duplicates(), on=["season", "round"], how="left") if "circuit" not in X else X
last = res2.merge(X[["season", "round", "k"]].drop_duplicates(), on=["season", "round"]).sort_values("k")
last["prev_pod_here"] = last.groupby(["circuit", "code"]).pod.shift(1); last["prev_team_pod_here"] = last.groupby(["circuit", "team"]).pod.shift(1)
X = X.merge(last[["season", "round", "code", "prev_pod_here", "prev_team_pod_here"]], on=["season", "round", "code"], how="left")
# standings variants (decayed points): different windows, the baseline family
for hl in (2, 4, 8):
    X["_p"] = X.points / 25; add(f"dpts_hl{hl}", X, "_p", "code", hl)
# Elo (team + driver additive), chronological, strengths recorded BEFORE each race
def elo(Kt, Kd, shrink, scale=400.0):
    Rt, Rd, out, prev = {}, {}, {}, None
    for k, g in X.sort_values("k").groupby("k", sort=True):
        s = g.season.iloc[0]
        if prev is not None and s != prev:
            for d in (Rt, Rd):
                for key in d: d[key] *= shrink
        prev = s
        S = np.array([Rt.get(t, 0) + Rd.get(c, 0) for t, c in zip(g.team, g.code)]); out[k] = pd.Series(S, index=g.index)
        o = g.order.values.astype(float); n = len(g)
        Eij = 1 / (1 + 10 ** (-(S[:, None] - S[None, :]) / scale)); Sij = (o[None, :] > o[:, None]).astype(float) + 0.5 * (o[None, :] == o[:, None])
        np.fill_diagonal(Eij, 0); np.fill_diagonal(Sij, 0); delta = (Sij - Eij).sum(1) / (n - 1)
        for t, c, dl in zip(g.team, g.code, delta):
            Rt[t] = Rt.get(t, 0) + Kt * dl / 2; Rd[c] = Rd.get(c, 0) + Kd * dl
    return pd.concat(out.values())
best = None
for Kt, Kd, sh in itertools.product((40, 100, 200), (10, 30), (0.5, 0.8)):
    X["_elo"] = elo(Kt, Kd, sh); rr = X[X.season.between(2020, 2022)].groupby("k").apply(lambda g: W.metrics(g.assign(gridn=0), -g._elo.values, tie=None)["podium"], include_groups=False).mean()
    if best is None or rr > best[0]: best = (rr, Kt, Kd, sh)
print(f"E33 Elo grid (chosen on INNER 2020-22): best inner podium {best[0]:.3f} at Kt={best[1]} Kd={best[2]} shrink={best[3]}")
X["elo"] = elo(*best[1:])
W.MODELS["E33_elo"] = lambda tr, te, F, **kw: -te.elo.values
# ---------- label-side
def lab_pace(tr, te, F, **kw):   # label = would-be podium by TRUE RACE PACE (immune to DNF/strategy luck); evaluated on real podium
    t = tr.dropna(subset=["y_pace"]).copy(); t["y"] = (t.groupby("k").y_pace.rank() <= 3).astype(int)
    return -pipe().fit(t[BASE], t.y, logisticregression__sample_weight=W._w(t, 100)).predict_proba(te[BASE])[:, 1]
def ridge_t(y):
    def f(tr, te, F, **kw):
        t = tr.copy(); t["y"] = y(t); return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=300)).fit(t[BASE], t.y, ridge__sample_weight=W._w(t, 100)).predict(te[BASE])
    return f
def dnf_down(tr, te, F, w=0.3, **kw):
    t = tr.copy(); wt = W._w(t, 100) * np.where(t.dnf_y == 1, w, 1.0); return -pipe().fit(t[BASE], t.podium, logisticregression__sample_weight=wt).predict_proba(te[BASE])[:, 1]
def soft(tr, te, F, **kw):      # soft label: podium=1, P4-5 = 0.4 (via duplicated weighted rows)
    t = tr.copy(); y = np.where(t.order <= 3, 1.0, np.where(t.order <= 5, 0.4, 0.0)); T = pd.concat([t, t]); Y = np.r_[np.ones(len(t)), np.zeros(len(t))]; Wt = np.r_[y, 1 - y] * np.r_[W._w(t, 100), W._w(t, 100)]
    return -pipe().fit(T[BASE], Y, logisticregression__sample_weight=Wt).predict_proba(te[BASE])[:, 1]
W.MODELS.update(E34_pace_label=lab_pace, E35_logorder_reg=ridge_t(lambda t: np.log(t.order)), E36_points_reg=lambda tr, te, F, **kw: -ridge_t(lambda t: np.log1p(t.points))(tr, te, F), E37_dnf_downweight=dnf_down, E38_soft_labels=soft)
# ---------- feature-side (logit on BASE + block)
blocks = {"E39_trend": ["t_trend_pace", "d_trend_pace", "t_trend_ord", "d_trend_ord"], "E40_q1gap_hist": ["d4_q1gap", "t4_q1gap", "d10_q1gap", "t10_q1gap"],
          "E41_start_skill": ["d10_lap1gain"], "E42_laps_led": ["d10_led", "t10_led", "t4_led"], "E43_sprint_form": ["d10_sprint"],
          "E45_circuit_continuity": ["prev_pod_here", "prev_team_pod_here"]}
for nm, ex in blocks.items(): W.MODELS[nm] = (lambda ex: lambda tr, te, F, **kw: W.m_logit_pod(tr, te, BASE + ex, C=0.03, hl=100))(ex)
W.MODELS["E44_decayed_points_hl2"] = lambda tr, te, F, **kw: -te.dpts_hl2.fillna(0).values; W.MODELS["E44_decayed_points_hl4"] = lambda tr, te, F, **kw: -te.dpts_hl4.fillna(0).values; W.MODELS["E44_decayed_points_hl8"] = lambda tr, te, F, **kw: -te.dpts_hl8.fillna(0).values
W.MODELS["E46_all_new_features"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, BASE + sum(blocks.values(), []), C=0.03, hl=100)
ms = ["standings", "logit"] + [m for m in W.MODELS if m.startswith("E")]
R, P = W.walk_forward(X, DEV, ms, ens={"ens": ["standings", "logit"]}, tie=None); report(R, "ens"); P.to_pickle("data/cache_P_r3.pkl")
