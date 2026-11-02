import sys; sys.path.insert(0, ".")
from experiments.common import *
X = table(); wx = pd.read_csv(DATA / "weather.csv"); X = X.merge(wx, on=["season", "round"], how="left")
BASE = pre_weekend_cols(X); DEV = [2020, 2021, 2022, 2023, 2024, 2025]
# sanity: does rain relate to chaos?
d = X.assign(dnfx=lambda t: t.d10_dnf).drop_duplicates("k")
res = load_results(); ch = res.assign(chaos=((res.order <= 3) & (res.grid > 3))).groupby(["season", "round"]).chaos.mean().rename("chaos").reset_index()
t = X.drop_duplicates("k")[["season", "round", "rain_act"]].merge(ch, on=["season", "round"])
t["wet"] = t.rain_act > 1.0
print("share of races with >1mm rain 12-17h local: %.2f ; podium-from-outside-top3-grid rate: wet %.3f vs dry %.3f (n=%d wet)" % (t.wet.mean(), t[t.wet].chaos.mean(), t[~t.wet].chaos.mean(), t.wet.sum()))
# wet skill: past-only decayed shrunk mean of (form-rank minus finish) in earlier wet races
X["resid"] = X.d10_order - X.order; X["wetrace"] = (X.rain_act > 1.0).astype(float)
skill = {}; ws = []
for k, g in X.groupby("k", sort=True):
    ws.append(pd.Series([skill.get(c, (0.0, 0.0)) for c in g.code], index=g.index).apply(lambda v: v[0] / (v[1] + 3)))
    if g.wetrace.iloc[0] == 1:
        for c, r in zip(g.code, g.resid):
            if np.isfinite(r): a, n = skill.get(c, (0.0, 0.0)); skill[c] = (a + r, n + 1)
X["d_wet_skill"] = pd.concat(ws)
X["rain_x_skill"] = X.rain_act * X.d_wet_skill; X["rain_x_form"] = X.rain_act * X.d10_order.fillna(11); X["rain_x_gridless"] = X.rain_act * X.field
W.MODELS["standings"] = lambda tr, te, F, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
for nm, extra in {"base": [], "+ORACLE rain": ["rain_act"], "+ORACLE rain,temp,wind": ["rain_act", "temp_act", "wind_act"],
                  "+ORACLE rain + wet skill": ["rain_act", "d_wet_skill", "rain_x_skill", "rain_x_form"]}.items():
    F = BASE + extra
    W.MODELS[f"lg|{nm}"] = lambda tr, te, F_, F=F, **kw: W.m_logit_pod(tr, te, F, C=0.03, hl=100)
    W.MODELS[f"gb|{nm}"] = lambda tr, te, F_, F=F, **kw: W.m_lgb_pod(tr, te, F, hl=100)
s, R, P = run(X, ["standings"] + [m for m in W.MODELS if "|" in m], DEV)
print(s.sort_index().to_string())
