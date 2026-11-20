"""E253 exact synthetic nonidentifiability, not a fitted F1 model."""
import json
import math
from fractions import Fraction as Q
from statistics import median

from f1.data import ROOT

DRIVERS={"A1":Q(-3,200),"A2":Q(-1,200),"B1":Q(1,200),"B2":Q(3,200)}
CARS={"A":Q(1,50),"B":Q(-1,50)}
BASES={1:Q(23,5),2:Q(47,10)}
ASSIGNMENTS=[(r,d,d[0]) for r in BASES for d in DRIVERS]


def observations(drivers,cars,assignments=ASSIGNMENTS):
    return {(r,d):BASES[r]+drivers[d]+cars[t] for r,d,t in assignments}


def shifted(drivers,cars,offsets):
    return ({d:v+offsets[d[0]] for d,v in drivers.items()},
            {t:v-offsets[t] for t,v in cars.items()})


def signals(observed):
    lap={(r,d):math.exp(float(v)) for (r,d),v in observed.items()}
    normalized={};field={}
    for r in BASES:
        mid=median(lap[(r,d)] for d in DRIVERS)
        for d in DRIVERS:
            mate=d[0]+("2" if d.endswith("1") else "1")
            normalized[(r,d)]=(lap[(r,d)]-lap[(r,mate)])/lap[(r,mate)]
            field[(r,d)]=(lap[(r,d)]-mid)/mid
    return normalized,field


def probability(drivers,a,b):
    return 1/(1+math.exp(100*float(drivers[a]-drivers[b])))


def main():
    alternative,cars=shifted(DRIVERS,CARS,{"A":Q(1,25),"B":Q(-1,25)})
    observed=observations(DRIVERS,CARS);other=observations(alternative,cars)
    original_signals=signals(observed);other_signals=signals(other)
    first_pairs=[probability(DRIVERS,a,b) for a,b in (("A1","A2"),("B1","B2"))]
    other_pairs=[probability(alternative,a,b) for a,b in (("A1","A2"),("B1","B2"))]
    before_bridge=Q(24,5)+DRIVERS["B1"]+CARS["A"]
    after_bridge=Q(24,5)+alternative["B1"]+cars["A"]
    result=dict(observations_equal_exactly=observed==other,
        teammate_normalized_signals_equal=original_signals[0]==other_signals[0],
        field_normalized_signals_equal=original_signals[1]==other_signals[1],
        centered_driver_and_car_constraints_preserved=all(sum(x.values())==0 for x in (DRIVERS,CARS,alternative,cars)),
        within_component_probabilities_equal=first_pairs==other_pairs,
        within_component_probabilities=first_pairs,
        world1_best_same_car_driver=min(DRIVERS,key=DRIVERS.get),world2_best_same_car_driver=min(alternative,key=alternative.get),
        cross_component_probability_world1=probability(DRIVERS,"A1","B1"),
        cross_component_probability_world2=probability(alternative,"A1","B1"),
        actual_teammate_bridge_breaks_invariance=before_bridge!=after_bridge,
        worlds={"world1":dict(driver={k:str(v) for k,v in DRIVERS.items()},car={k:str(v) for k,v in CARS.items()}),
                "world2":dict(driver={k:str(v) for k,v in alternative.items()},car={k:str(v) for k,v in cars.items()})},
        notebook_commit="4ab7ae5379bff1f4fa5b616bd629d2e1192981f8",notebook_blob="1948a6cab72effafc9a37ee8ba9364d695144871",
        real_race_evaluated=False,restricted_data_ingested=False,winner_model_fitted=False)
    assert all(result[k] for k in ("observations_equal_exactly","teammate_normalized_signals_equal","field_normalized_signals_equal",
        "centered_driver_and_car_constraints_preserved","within_component_probabilities_equal","actual_teammate_bridge_breaks_invariance"))
    assert result["world1_best_same_car_driver"]!=result["world2_best_same_car_driver"]
    (ROOT/"results/e253_skill_identifiability.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
