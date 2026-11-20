from fractions import Fraction as Q
from experiments.e253_skill_identifiability import DRIVERS,CARS,observations,shifted,signals,probability


def main():
    d,c=shifted(DRIVERS,CARS,{"A":Q(1,25),"B":Q(-1,25)})
    assert observations(DRIVERS,CARS)==observations(d,c)
    assert observations(DRIVERS,CARS)!=observations(d,CARS)
    assert all(sum(x.values())==0 for x in (DRIVERS,CARS,d,c))
    assert min(DRIVERS,key=DRIVERS.get)=="A1" and min(d,key=d.get)=="B1"
    assert signals(observations(DRIVERS,CARS))==signals(observations(d,c))
    assert probability(DRIVERS,"A1","A2")==probability(d,"A1","A2")
    assert probability(DRIVERS,"A1","B1")!=probability(d,"A1","B1")
    # Independent design-null-vector proof includes both zero-sum constraints.
    direction=[1,1,-1,-1,-1,1]
    rows=[[1,0,0,0,1,0],[0,1,0,0,1,0],[0,0,1,0,0,1],[0,0,0,1,0,1],
          [1,1,1,1,0,0],[0,0,0,0,1,1]]
    assert all(sum(a*b for a,b in zip(row,direction))==0 for row in rows)
    bridge=[0,0,1,0,1,0]
    assert sum(a*b for a,b in zip(bridge,direction))!=0
    assert DRIVERS["B1"]-DRIVERS["A1"]!=d["B1"]-d["A1"]
    from f1.data import ROOT
    import json
    artifact=json.loads((ROOT/"results/e253_skill_identifiability.json").read_text())
    for world in ("world1","world2"):
        assert sum(Q(v) for v in artifact["worlds"][world]["driver"].values())==0
        assert sum(Q(v) for v in artifact["worlds"][world]["car"].values())==0
    assert artifact["world1_best_same_car_driver"]=="A1" and artifact["world2_best_same_car_driver"]=="B1"
    assert all(artifact[k] for k in ("observations_equal_exactly","teammate_normalized_signals_equal","field_normalized_signals_equal", "centered_driver_and_car_constraints_preserved","within_component_probabilities_equal","actual_teammate_bridge_breaks_invariance"))
    assert not any(artifact[k] for k in ("real_race_evaluated","restricted_data_ingested","winner_model_fitted"))
    print("E253 exact observational/centered/normalized invariance, negative control and independent design-null/bridge checks PASS")


if __name__=="__main__":main()
