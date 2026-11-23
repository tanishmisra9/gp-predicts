"""E268 train-only inputs, ordered model guards and real source poison checks."""
import numpy as np
import pandas as pd

from experiments.e268_ordered_winner import CONFIG, CATS, fit_score, inputs, labels, make_features
from f1.data import DATA, load_results
from f1.features import pre_weekend_cols


def fixture():
    rows = [dict(k=k, driver=f"d{i}", team=f"t{i//2}", circuit_id=f"c{k%3}",
                 date=pd.Timestamp("2019-01-06")+pd.Timedelta(days=7*k), win=int(i==0),
                 history=float(i), missing=np.nan) for k in range(40) for i in range(4)]
    train = pd.DataFrame(rows)
    current = train.iloc[-4:].copy().assign(k=40, date=pd.Timestamp("2020-01-05"), circuit_id="newΩ")
    current.iloc[0, current.columns.get_loc("team")] = None
    return train, current


def synthetic():
    train, current = fixture(); F = ["history", "missing"]
    prior, a, b = inputs(train, current, F)
    assert b.team.iloc[0] == "__missing__" and b.circuit_id.eq("newΩ").all()
    assert "newΩ" not in set(a.circuit_id)
    y = labels(prior, True)
    prefix = prior[prior.k.lt(20)]
    np.testing.assert_array_equal(y[:len(prefix)], labels(prefix, True))
    assert pd.Series(y).groupby(prior.k).sum().eq(1).all()
    for bad in [train.iloc[:0], pd.concat([train, current]), pd.concat([train, train.iloc[:1]])]:
        try: inputs(bad, current, F)
        except AssertionError: pass
        else: raise AssertionError("bad training prefix accepted")
    try: inputs(train, current.assign(gridn=1), F+["gridn"])
    except ValueError: pass
    else: raise AssertionError("forbidden feature accepted")
    for kwargs in [{}, dict(boosting="Plain"), dict(categorical=False), dict(shuffled=True)]:
        first, params = fit_score(train, current, F, **kwargs)
        other, _ = fit_score(train.sample(frac=1, random_state=1), current.assign(win=999, order=-1, gridn=-999), F, **kwargs)
        np.testing.assert_array_equal(first, other)
        assert params["has_time"] and params["boosting_type"] == kwargs.get("boosting", "Ordered")
        for key in ["iterations", "depth", "max_ctr_complexity", "counter_calc_method"]:
            if key in params:
                assert params[key] == CONFIG[key]
    extra = current.iloc[:1].copy().assign(driver="unseen追加", team="unseen", circuit_id="other", history=1e9)
    before, _ = fit_score(train, current, F)
    after, _ = fit_score(train, pd.concat([current, extra]), F)
    np.testing.assert_array_equal(before, after[:len(before)])
    print("E268 native NaN/Unicode/unseen categories, prefix-null, deterministic guards and target-batch invariance PASS")


def real_source():
    results = load_results().loc[lambda d: d.season.le(2022)]
    pace = pd.read_csv(DATA / "racepace_ff1.csv").loc[lambda d: d.season.le(2022)]
    X = make_features(results, pace); F = pre_weekend_cols(X)
    target = X.season.eq(2022) & X["round"].eq(10); k = X.loc[target, "k"].iloc[0]
    changed = results.copy(); future = changed.season.eq(2022) & changed["round"].ge(10)
    for col in ["grid", "q_pos", "s_grid", "s_order", "order", "finish", "points", "laps"]:
        changed.loc[future, col] = 999
    for col in ["q1", "q2", "q3"]:
        changed.loc[future, col] = 999.
    changed.loc[future, "status"] = "poison"
    later = changed.season.eq(2022) & changed["round"].gt(10)
    for col in ["driver", "team", "circuit"]:
        changed.loc[later, col] = changed.loc[later, col].astype(str)+"futureΩ"
    changed_pace = pace.copy(); later_pace = changed_pace.season.eq(2022) & changed_pace["round"].ge(10)
    changed_pace.loc[later_pace, "pace_gap"] = 999.
    Z = make_features(changed, changed_pace)
    columns = F+CATS+["date"]
    pd.testing.assert_frame_equal(X.loc[X.k.le(k), columns], Z.loc[Z.k.le(k), columns])
    prefix_results = results[results.season.lt(2022) | (results.season.eq(2022) & results["round"].le(10))]
    prefix_pace = pace[pace.season.lt(2022) | (pace.season.eq(2022) & pace["round"].le(10))]
    prefix_X = make_features(prefix_results, prefix_pace)
    pd.testing.assert_frame_equal(X.loc[X.k.le(k), columns], prefix_X[columns])
    train = X.k.lt(k) & X.season.ge(2019)
    original = None
    for kwargs in [{}, dict(boosting="Plain"), dict(categorical=False), dict(shuffled=True)]:
        a, _ = fit_score(X[train], X[target], F, **kwargs)
        b, _ = fit_score(Z[train], Z[target], F, **kwargs)
        np.testing.assert_array_equal(a, b)
        if not kwargs:
            original = a
    past_changed = results.copy(); prior = past_changed.season.eq(2022) & past_changed["round"].eq(9)
    past_changed.loc[prior, "order"] = int(prior.sum())+1-past_changed.loc[prior, "order"]
    past_changed.loc[prior, "points"] += 100
    earlier = make_features(past_changed, pace)
    assert not earlier.loc[target, F].equals(X.loc[target, F])
    responsive, _ = fit_score(earlier[train], earlier[target], F)
    assert not np.array_equal(responsive, original)
    print("E268 real current/future result+grid+quali+sprint+pace+future identities/prefix invariance and earlier fitted response PASS")


if __name__ == "__main__":
    synthetic(); real_source()
