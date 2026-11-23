import json
from experiments.e261_creator_audit import skip_policy,toy
from f1.data import ROOT


def main():
    source="""def walk_forward():
    for race in races:
        if not alignment.scoreable:
            skipped.append({"reason":"winner_not_named"})
            continue
"""
    assert skip_policy(source)
    assert not skip_policy(source.replace("not alignment.scoreable","alignment.scoreable"))
    assert not skip_policy(source.replace("alignment.scoreable","other.scoreable"))
    assert not skip_policy(source.replace("continue","pass"))
    assert not skip_policy(source.replace("winner_not_named","other"))
    assert toy()["conditional_accuracy"]>toy()["all_race_accuracy"]
    saved=json.loads((ROOT/"results/e261_creator_audit.json").read_text())
    assert saved["winner_not_named_skip_present"] and not saved["forecast_replicated"]
    assert not saved["external_code_executed"] and not saved["proves_universal_accuracy_ceiling"]
    assert saved["reported_metrics"]["strength_mc"]["hits"]==saved["reported_metrics"]["championship"]["hits"]==43
    print("E261 policy AST mutation/denominator counterexample/limited claims PASS")


if __name__=="__main__":main()
