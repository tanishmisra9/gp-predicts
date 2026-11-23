"""E268 inner-only winner test; optional dependencies: requirements-research.txt."""
import hashlib
import json

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.e168_winner import win_score
from experiments.e172_nonlinear_winner import winner_gbm
from experiments.e201_categorical_winner import CATS, attach_context
from f1.data import DATA, ROOT, load_results
from f1.features import build_features, pre_weekend_cols

CONFIG = dict(loss_function="Logloss", has_time=True, iterations=250, depth=3,
              learning_rate=.03, l2_leaf_reg=5, bootstrap_type="Bernoulli", subsample=.8,
              rsm=.7, random_seed=0, thread_count=1, task_type="CPU", grow_policy="SymmetricTree",
              max_ctr_complexity=1, one_hot_max_size=2, counter_calc_method="SkipTest",
              nan_mode="Min", allow_writing_files=False, verbose=False)
CONTROLS = ["plain_ids", "ordered_numeric", "gbm_base", "win_logit", "standings", "ens"]


def inputs(train, current, features, categorical=True):
    W.assert_pre_weekend(features + (CATS if categorical else []))
    assert not set(features) & set(CATS)
    assert len(train) and len(current) and current.k.nunique() == 1
    assert train.k.max() < current.k.min()
    assert not train.duplicated(["k", "driver"]).any() and not current.driver.duplicated().any()
    assert current.date.nunique() == 1
    date = pd.Timestamp(current.date.iloc[0]).normalize()
    cutoff = date - pd.Timedelta(days=(date.weekday()-3) % 7)
    assert train.date.lt(cutoff).all()
    prior = train.sort_values(["k", "driver"]).reset_index(drop=True)
    a = prior[features].astype(float).copy(); b = current[features].astype(float).copy()
    assert not np.isinf(a.to_numpy()).any() and not np.isinf(b.to_numpy()).any()
    if categorical:
        for name in CATS:
            a[name] = prior[name].fillna("__missing__").astype(str)
            b[name] = current[name].fillna("__missing__").astype(str)
    return prior, a, b


def labels(prior, shuffled=False):
    y = prior.win.to_numpy().copy()
    assert set(y) <= {0, 1} and prior.groupby("k").win.sum().eq(1).all()
    if shuffled:
        y[:] = 0; rng = np.random.default_rng(268)
        for indices in prior.groupby("k", sort=True).indices.values():
            y[rng.choice(indices)] = 1
    assert y.sum() == prior.k.nunique()
    return y


def fit_score(train, current, features, categorical=True, boosting="Ordered", shuffled=False):
    import catboost
    assert catboost.__version__ == "1.2.10" and boosting in {"Ordered", "Plain"}
    prior, a, b = inputs(train, current, features, categorical)
    model = catboost.CatBoostClassifier(**CONFIG, boosting_type=boosting)
    model.fit(a, labels(prior, shuffled), cat_features=CATS if categorical else [],
              sample_weight=W._w(prior, 100))
    probability = model.predict_proba(b, thread_count=1)[:, 1]
    assert len(probability) == len(current) and np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all() and model.tree_count_ == 250
    return -probability, model.get_all_params()


def make_features(results, pace):
    keys = ["season", "round", "code"]
    assert results.season.le(2022).all() and pace.season.le(2022).all()
    assert not results.duplicated(keys).any()
    return attach_context(build_features(results, pace), results).merge(
        results[keys+["date"]], on=keys, validate="one_to_one")


def main():
    assert not (ROOT / "results/e268_inner_predictions.csv").exists(), "do not overwrite a scored run"
    results = load_results().loc[lambda d: d.season.le(2022)]
    pace = pd.read_csv(DATA / "racepace_ff1.csv").loc[lambda d: d.season.le(2022)]
    X = make_features(results, pace); F = pre_weekend_cols(X)
    W.assert_pre_weekend(F+CATS)
    for _, target in X[X.season.between(2020, 2022)].groupby("k"):
        assert len(X[X.k.lt(target.k.iloc[0]) & X.season.ge(2019)])
    configurations = {}; fits = []

    def run(train, current, features, name, **kwargs):
        scores, params = fit_score(train, current, features, **kwargs)
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        configurations[key] = params
        fits.append(dict(season=int(current.season.iloc[0]), round=int(current["round"].iloc[0]),
                         model=name, configuration=key, training_races=int(train.k.nunique())))
        print("FIT", fits[-1]["season"], fits[-1]["round"], name, flush=True)
        return scores

    W.MODELS.update(
        standings=lambda tr, te, F: (-te.d_season_pts.fillna(0)*100 + te.d10_order.fillna(11)*1e-3).to_numpy(),
        pre_logit=lambda tr, te, F: W.m_logit_pod(tr, te, F, C=.03, hl=100),
        win_logit=lambda tr, te, F: win_score(tr, te, F), gbm_base=winner_gbm,
        ordered_ids=lambda tr, te, F: run(tr, te, F, "ordered_ids"),
        plain_ids=lambda tr, te, F: run(tr, te, F, "plain_ids", boosting="Plain"),
        ordered_numeric=lambda tr, te, F: run(tr, te, F, "ordered_numeric", categorical=False),
        ordered_null=lambda tr, te, F: run(tr, te, F, "ordered_null", shuffled=True))
    names = ["standings", "pre_logit", "win_logit", "gbm_base", "ordered_ids", "plain_ids", "ordered_numeric", "ordered_null"]
    R, P = W.walk_forward(X, [2020, 2021, 2022], names, F=F,
                          ens={"ens": ["standings", "pre_logit"]}, tie=None)
    assert R.groupby("model").size().eq(61).all() and P.groupby("model").size().eq(1220).all()
    old = pd.read_csv(ROOT / "results/e201_inner_predictions.csv", float_precision="round_trip")
    for name in ["gbm_base", "win_logit", "standings", "ens"]:
        check = P[P.model.eq(name)].merge(old[old.model.eq(name)][["season", "round", "driver", "score"]],
                                        on=["season", "round", "driver"], suffixes=("_new", "_old"), validate="one_to_one")
        assert len(check) == 1220
        np.testing.assert_allclose(check.score_new, check.score_old, atol=1e-14, rtol=0)
    truth = X[["season", "round", "driver", "team", "order", "win"]]
    P = P.merge(truth, on=["season", "round", "driver"], validate="many_to_one")
    probabilistic = set(names) - {"standings"}
    eligible = P.model.isin(probabilistic)
    positive = (-P.loc[eligible, "score"]).clip(lower=1e-12)
    P["p_win"] = np.nan
    P.loc[eligible, "p_win"] = positive / positive.groupby([P.loc[eligible, c] for c in ["season", "round", "model"]]).transform("sum")
    probability_rows = []
    for (season, round_, name), g in P[eligible].groupby(["season", "round", "model"], sort=True):
        probability_rows.append(dict(season=int(season), round=int(round_), model=name,
                                     logloss=float(-np.log(g.loc[g.win.eq(1), "p_win"]).sum()),
                                     brier=float(((g.p_win-g.win)**2).sum())))
    probability_rows = pd.DataFrame(probability_rows)
    models = {}
    winners = truth[truth.win.eq(1)].set_index(["season", "round"]).team
    picked = P.sort_values(["score", "driver"]).drop_duplicates(["season", "round", "model"])
    for name, rows in R.groupby("model"):
        chosen = picked[picked.model.eq(name)]
        models[name] = dict(winners=int(rows.winner.sum()), winner_rate=float(rows.winner.mean()),
                            overlap=float(rows.podium.mean()), by_season=rows.groupby("season").winner.sum().astype(int).to_dict(),
                            team_hits=sum(team == winners.loc[(s, r)] for s, r, team in chosen[["season", "round", "team"]].itertuples(index=False, name=None)))
        if name in probabilistic:
            models[name].update(probability_rows[probability_rows.model.eq(name)][["logloss", "brier"]].mean().to_dict())
    pivot = R.pivot(index=["season", "round"], columns="model", values="winner").sort_index()
    paired = {}
    for name in CONTROLS:
        diff, ci, p = W.paired_boot(pivot.ordered_ids, pivot[name])
        paired[name] = dict(diff=float(diff), ci=ci.tolist(), p_nonpositive=float(p))
    summary = dict(models=models, paired=paired, candidate="ordered_ids", gate=bool(
        models["ordered_ids"]["winners"] >= 37 and all(p["ci"][0] > 0 for p in paired.values())
        and models["ordered_ids"]["winners"] > models["ordered_null"]["winners"]),
        package_version="1.2.10", configurations=configurations, fits=fits,
        features=F, categorical_features=CATS, outer_evaluated=False, evaluated_2026=False,
        evaluated_races=61, forecast_entrants=1220,
        source_hashes={name: hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
                       for name, frame in [("results_through_2022", results), ("racepace_through_2022", pace)]})
    R.to_csv(ROOT / "results/e268_inner_per_race.csv", index=False)
    P.to_csv(ROOT / "results/e268_inner_predictions.csv", index=False)
    probability_rows.to_csv(ROOT / "results/e268_inner_probability_metrics.csv", index=False)
    summary["artifact_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (ROOT / "results").glob("e268_inner_*.csv")}
    (ROOT / "results/e268_inner_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print("RESULT", json.dumps(dict(models=models, paired=paired, gate=summary["gate"])), flush=True)


if __name__ == "__main__":
    main()
