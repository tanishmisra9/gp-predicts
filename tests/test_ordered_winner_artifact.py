"""E268 official-label ranking, probability, bootstrap and direct-fit verification."""
import hashlib
import importlib.metadata
import json
import math

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from experiments.e268_ordered_winner import make_features
from f1.data import DATA, ROOT, load_results
from f1.features import pre_weekend_cols


def main():
    result = json.loads((ROOT / "results/e268_inner_summary.json").read_text())
    P = pd.read_csv(ROOT / "results/e268_inner_predictions.csv", float_precision="round_trip")
    R = pd.read_csv(ROOT / "results/e268_inner_per_race.csv", float_precision="round_trip")
    C = pd.read_csv(ROOT / "results/e268_inner_probability_metrics.csv", float_precision="round_trip")
    results = load_results().loc[lambda d: d.season.le(2022)]
    pace = pd.read_csv(DATA / "racepace_ff1.csv").loc[lambda d: d.season.le(2022)]
    truth = results[results.season.between(2020, 2022)].set_index(["season", "round", "driver"])
    assert len(P) == 1220*9 and len(R) == 61*9 and len(C) == 61*7
    assert R.groupby("model").size().eq(61).all() and P.groupby("model").size().eq(1220).all()
    assert set(result["paired"]) == {"plain_ids", "ordered_numeric", "gbm_base", "win_logit", "standings", "ens"}
    assert not P.duplicated(["season", "round", "model", "driver"]).any()
    assert result["evaluated_races"] == 61 and result["forecast_entrants"] == 1220
    assert set(P.season) == {2020, 2021, 2022} and not result["outer_evaluated"] and not result["evaluated_2026"]
    observed = {}; team_hits = {}; by_season = {}; losses = {}
    for (season, round_, model), g in P.groupby(["season", "round", "model"], sort=True):
        event = truth.loc[(season, round_)]
        assert set(g.driver) == set(event.index)
        ranking = sorted(zip(g.score, g.driver), key=lambda row: (row[0], row[1]))
        predicted = [driver for _, driver in ranking]
        actual = list(event.sort_values("order").index)
        hit = int(predicted[0] == actual[0]); overlap = len(set(predicted[:3]) & set(actual[:3]))/3
        row = R[R.season.eq(season) & R["round"].eq(round_) & R.model.eq(model)].iloc[0]
        assert row.winner == hit and np.isclose(row.podium, overlap)
        assert row.exact_podium == (set(predicted[:3]) == set(actual[:3]))
        for n in [5, 10]:
            assert np.isclose(row[f"top{n}"], len(set(predicted[:n]) & set(actual[:n]))/n)
        assert np.isclose(row.mae, sum(abs(i+1-event.loc[d, "order"]) for i, d in enumerate(predicted))/len(predicted))
        gains = {d: 3-i for i, d in enumerate(actual[:3])}
        ndcg = sum(gains.get(d, 0)/math.log2(i+2) for i, d in enumerate(predicted[:3]))/sum((3-i)/math.log2(i+2) for i in range(3))
        assert np.isclose(row.ndcg3, ndcg)
        observed.setdefault(model, []).append(hit)
        by_season[(model, season)] = by_season.get((model, season), 0)+hit
        team_hits[model] = team_hits.get(model, 0)+int(event.loc[predicted[0], "team"] == event.loc[actual[0], "team"])
        for d in g.itertuples():
            assert d.order == event.loc[d.driver, "order"] and d.win == int(d.order == 1)
        if model not in {"standings", "ens"}:
            positive = [max(-float(score), 1e-12) for score in g.score]
            total = math.fsum(positive); probability = [p/total for p in positive]
            np.testing.assert_allclose(g.p_win, probability, rtol=1e-13, atol=1e-13)
            ll = -math.log(probability[list(g.driver).index(actual[0])])
            brier = math.fsum((p-int(driver == actual[0]))**2 for p, driver in zip(probability, g.driver))
            saved = C[C.season.eq(season) & C["round"].eq(round_) & C.model.eq(model)].iloc[0]
            np.testing.assert_allclose([saved.logloss, saved.brier], [ll, brier], rtol=1e-13, atol=1e-13)
            losses.setdefault(model, []).append([ll, brier])
    for model, values in observed.items():
        summary = result["models"][model]
        assert summary["winners"] == sum(values) and summary["winner_rate"] == sum(values)/61
        assert summary["team_hits"] == team_hits[model]
        assert np.isclose(summary["overlap"], R[R.model.eq(model)].podium.mean())
        for year in [2020, 2021, 2022]:
            assert summary["by_season"][str(year)] == by_season[(model, year)]
        if model in losses:
            np.testing.assert_allclose([summary["logloss"], summary["brier"]], np.mean(losses[model], axis=0), rtol=1e-13, atol=1e-13)
    candidate = np.array(observed["ordered_ids"])
    for model, saved in result["paired"].items():
        other = np.array(observed[model]); difference = candidate-other
        indices = np.random.default_rng(0).integers(0, 61, size=(5000, 61))
        draws = (candidate[indices]-other[indices]).mean(axis=1)
        assert saved["diff"] == difference.mean() and saved["p_nonpositive"] == (draws <= 0).mean()
        np.testing.assert_allclose(saved["ci"], np.percentile(draws, [2.5, 97.5]), rtol=0, atol=1e-15)
    gate = sum(candidate)>=37 and all(p["ci"][0]>0 for p in result["paired"].values()) and sum(candidate)>sum(observed["ordered_null"])
    assert result["gate"] == gate and result["candidate"] == "ordered_ids"
    assert len(result["fits"]) == 61*4
    assert len({(f["season"], f["round"], f["model"]) for f in result["fits"]}) == 61*4
    assert {f["model"] for f in result["fits"]} == {"ordered_ids", "plain_ids", "ordered_numeric", "ordered_null"}
    earlier_events = list(results[results.season.ge(2019)][["season", "round"]].drop_duplicates().itertuples(index=False, name=None))
    for f in result["fits"]:
        assert f["season"] in {2020, 2021, 2022}
        assert f["training_races"] == sum(event < (f["season"], f["round"]) for event in earlier_events)
        config = result["configurations"][f["configuration"]]
        assert hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() == f["configuration"]
        assert config["iterations"] == 250 and config["depth"] == 3 and config["has_time"]
        assert config["boosting_type"] == ("Plain" if f["model"] == "plain_ids" else "Ordered")
        assert config["random_seed"] == 0 and config["task_type"] == "CPU" and not config["use_best_model"]
        assert config["grow_policy"] == "SymmetricTree" and config["loss_function"] == "Logloss" and config["nan_mode"] == "Min"
        for key, expected in [("learning_rate", .03), ("l2_leaf_reg", 5), ("subsample", .8), ("rsm", .7)]:
            assert np.isclose(config[key], expected, rtol=1e-6, atol=0)
        if f["model"] != "ordered_numeric":
            assert config["max_ctr_complexity"] == 1 and config["counter_calc_method"] == "SkipTest"
    for name, frame in [("results_through_2022", results), ("racepace_through_2022", pace)]:
        assert hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest() == result["source_hashes"][name]
    for name, digest in result["artifact_hashes"].items():
        assert hashlib.sha256((ROOT / "results" / name).read_bytes()).hexdigest() == digest
    assert importlib.metadata.version("catboost") == result["package_version"] == "1.2.10"
    # Direct, separately constructed API fits for one fixed target, including regenerated null labels.
    X = make_features(results, pace); F = pre_weekend_cols(X)
    te = X[X.season.eq(2022) & X["round"].eq(10)]
    tr = X[X.season.ge(2019) & X.k.lt(te.k.iloc[0])].sort_values(["k", "driver"]).reset_index(drop=True)
    assert len(tr) and tr.date.lt(pd.Timestamp("2022-06-30")).all()
    fixed = dict(loss_function="Logloss", has_time=True, iterations=250, depth=3, learning_rate=.03,
                 l2_leaf_reg=5, bootstrap_type="Bernoulli", subsample=.8, rsm=.7, random_seed=0,
                 thread_count=1, task_type="CPU", grow_policy="SymmetricTree", max_ctr_complexity=1,
                 one_hot_max_size=2, counter_calc_method="SkipTest", nan_mode="Min", allow_writing_files=False, verbose=False)
    for model in ["ordered_ids", "plain_ids", "ordered_numeric", "ordered_null"]:
        cats = [] if model == "ordered_numeric" else ["driver", "team", "circuit_id"]
        a = tr[F].astype(float).copy(); b = te[F].astype(float).copy()
        for c in cats:
            a[c] = tr[c].fillna("__missing__").astype(str); b[c] = te[c].fillna("__missing__").astype(str)
        y = tr.win.to_numpy().copy()
        if model == "ordered_null":
            y[:] = 0; rng = np.random.default_rng(268)
            for k in sorted(tr.k.unique()):
                ids = np.flatnonzero(tr.k.eq(k).to_numpy()); y[ids[rng.integers(len(ids))]] = 1
            assert y.sum() == tr.k.nunique()
        fitted = CatBoostClassifier(**fixed, boosting_type="Plain" if model == "plain_ids" else "Ordered")
        fitted.fit(a, y, cat_features=cats, sample_weight=2.**(-(tr.k.max()-tr.k)/100))
        probability = fitted.predict_proba(b, thread_count=1)[:, 1]
        saved = P[P.season.eq(2022) & P["round"].eq(10) & P.model.eq(model)].set_index("driver").loc[te.driver]
        np.testing.assert_array_equal(-probability, saved.score.to_numpy())
    print("E268 ALL61official ranking/probability/season/bootstrap/config/hash/gate + direct4targetfits/null audit PASS")


if __name__ == "__main__":
    main()
