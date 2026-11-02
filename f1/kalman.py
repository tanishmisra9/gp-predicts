"""Local-level Kalman filter per team: latent pace level that drifts race to race (process var q) and jumps between seasons (extra var qs).
forecast(k) uses observations of races < k only (the filter is run chronologically and the one-step-ahead prediction is stored BEFORE the update).
Observations: team mean of a per-driver gap metric, centred on the field mean of that race (so values are relative pace)."""
import numpy as np
import pandas as pd


def team_obs(X, col):
    """team-weekend observation of `col` (post-race metric of that weekend), field-mean centred."""
    d = X.dropna(subset=[col])
    t = d.groupby(["k", "season", "round", "team"])[col].mean().reset_index()
    t[col] = t[col] - t.groupby("k")[col].transform("mean")
    return t


def run_kf(X, col, q=0.02, qs=0.5, r=0.5, prior_var=4.0, new_team_var=4.0):
    """returns DataFrame[k, team, f, P] = forecast mean/var for every team entered in race k, BEFORE seeing race k."""
    obs = team_obs(X, col).set_index(["k", "team"])[col].to_dict()
    teams_by_k = X.groupby("k").team.unique().to_dict(); season_of = X.drop_duplicates("k").set_index("k").season.to_dict()
    st = {}; out = []; prev_season = None
    for k in sorted(teams_by_k):
        s = season_of[k]
        for tm in teams_by_k[k]:
            if tm not in st:
                st[tm] = [0.0, new_team_var]
            m, P = st[tm]
            P = P + q + (qs if (prev_season is not None and s != prev_season) else 0.0)  # predict
            st[tm] = [m, P]; out.append((k, tm, m, P))
        for tm in teams_by_k[k]:  # update with this race's observation
            if (k, tm) in obs:
                m, P = st[tm]; K = P / (P + r); st[tm] = [m + K * (obs[(k, tm)] - m), (1 - K) * P]
        prev_season = s
    return pd.DataFrame(out, columns=["k", "team", "f", "P"])


def add_kf(X, col, name, **kw):
    f = run_kf(X, col, **kw).rename(columns={"f": name, "P": name + "_var"})
    return X.merge(f, on=["k", "team"], how="left")
