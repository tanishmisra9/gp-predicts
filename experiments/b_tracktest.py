import sys; sys.path.insert(0, ".")
from experiments.common import *
from f1.kalman import team_obs
from f1.profile import load_aux
from sklearn.linear_model import Ridge
X = table(); tel, circ, strat, ses, pits, _ = load_aux()
df = load_results()
cm = circ.merge(df.drop_duplicates(["season", "round"])[["season", "round", "circuit"]], on=["season", "round"])
cm["f_slow"] = cm.c_slow / (cm.c_slow + cm.c_med + cm.c_fast); cm["f_fast"] = cm.c_fast / (cm.c_slow + cm.c_med + cm.c_fast)
st = strat.merge(df.drop_duplicates(["season", "round"])[["season", "round", "circuit"]], on=["season", "round"])
T = cm.groupby("circuit")[["f_slow", "f_fast", "c_ft", "c_vmax", "c_lowspeed", "c_longest", "c_len"]].mean()
T = T.join(st.groupby("circuit")[["stops_2plus", "sc_frac"]].mean()); T = (T - T.mean()) / T.std(); T = T.fillna(0)
for col, label in (("y_pace", "RACE pace"), ("q_gap", "QUALI gap")):
    o = team_obs(X, col).merge(X.drop_duplicates("k")[["k", "circuit"]] if "circuit" in X else df.drop_duplicates(["season", "round"])[["season", "round", "circuit"]], on=["season", "round"] if "circuit" not in X else ["k"])
    o = o[o.season.between(2020, 2026) & o.circuit.isin(T.index)].copy()
    o["res"] = o[col] - o.groupby(["season", "team"])[col].transform("mean")           # deviation from the team's own season mean
    o["res"] -= o.groupby("k").res.transform("mean")
    for alpha in (5.0, 20.0, 80.0):
        pred = pd.Series(0.0, index=o.index)
        for (s, tm), g in o.groupby(["season", "team"]):
            if len(g) < 8: continue
            Z = T.loc[g.circuit].values
            for i in range(len(g)):                       # leave-one-race-out WITH hindsight on other races (incl. future)
                tr = np.ones(len(g), bool); tr[i] = False
                # no intercept: residuals are demeaned, a LOO intercept would anti-predict
                m = Ridge(alpha=alpha, fit_intercept=False).fit(Z[tr], g.res.values[tr])
                pred.loc[g.index[i]] = m.predict(Z[i:i + 1])[0]
        ok = pred != 0
        print(f"{label:10s} alpha={alpha:5.0f}: corr(predicted dev, actual dev) = {np.corrcoef(pred[ok], o.res[ok])[0, 1]:+.3f}  "
              f"R^2 vs zero = {1 - ((o.res[ok] - pred[ok]) ** 2).sum() / (o.res[ok] ** 2).sum():+.3f}  (n={ok.sum()} team-races)")
