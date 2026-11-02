import sys; sys.path.insert(0, ".")
from experiments.common import *
old = pd.read_csv(DATA / "results_2014_2017.csv", parse_dates=["date"]); df = pd.concat([old, load_results()], ignore_index=True)
pace = pd.read_csv(DATA / "racepace_ff1.csv")
X = build_features(df, pace); BASE = pre_weekend_cols(X)
print("races available:", X.k.nunique(), "| rows:", len(X))
W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
W.MODELS["logit"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, BASE, C=0.03, hl=100)
W.MODELS["ridge_pace"] = lambda tr, te, F, **kw: W.m_logit_pod(tr, te, BASE, C=0.01, hl=100)
for first in (2019, 2016, 2014):
    R, _ = W.walk_forward(X, [2020, 2021, 2022, 2023, 2024, 2025], ["standings", "logit"], ens={"ens": ["standings", "logit"]}, min_train_season=first, tie=None)
    r = R.assign(split=np.where(R.season <= 2022, "in", "out")).groupby(["model", "split"]).podium.mean().unstack().round(3)
    print(f"training from {first}:  logit in/out {r.loc['logit','in']:.3f}/{r.loc['logit','out']:.3f}   ens in/out {r.loc['ens','in']:.3f}/{r.loc['ens','out']:.3f}   standings {r.loc['standings','in']:.3f}/{r.loc['standings','out']:.3f}")
