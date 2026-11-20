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
    print("E253 exact observational/centered/normalized invariance, negative control and independent design-null/bridge checks PASS")


if __name__=="__main__":main()
