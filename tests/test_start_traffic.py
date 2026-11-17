"""E190 exposure is known at lap start; consecutive pairs share no lap."""
import numpy as np
import pandas as pd
from experiments.e190_traffic_feasibility import start_exposure,disjoint_pairs


def test_start_exposure():
    # B crossed ahead on a different lap count; A's own preceding crossing is excluded.
    x=pd.DataFrame(dict(Driver=['B','A','A','C'],Time=pd.to_timedelta([99.,100.,110.,130.],unit='s'),
        LapStartTime=pd.to_timedelta([89.,90.,100.,120.],unit='s'),LapTime=pd.to_timedelta([10.]*4,unit='s'),
        LapNumber=[7,2,3,4],Stint=[1]*4,Compound=['HARD']*4,TyreLife=[7.,2.,3.,4.],Position=[2.]*4,
        PitInTime=[pd.NaT]*4,PitOutTime=[pd.NaT]*4,TrackStatus=['1']*4,IsAccurate=[True]*4))
    x['PitInTime']=pd.to_timedelta(x.PitInTime)
    x['PitOutTime']=pd.to_timedelta(x.PitOutTime)
    got=start_exposure(x)
    target=got.Driver.eq('A')&got.LapNumber.eq(3)
    assert got.loc[target,'close'].iloc[0] and np.isclose(got.loc[target,'start_gap'].iloc[0],1.)
    changed=x.copy(); changed.loc[changed.Driver.eq('A')&changed.LapNumber.eq(3),'Time']=pd.Timedelta(seconds=119.)
    after=start_exposure(changed)
    after_target=after.Driver.eq('A')&after.LapNumber.eq(3)
    assert after.loc[after_target,'start_gap'].iloc[0]==got.loc[target,'start_gap'].iloc[0]
    self_only=start_exposure(x[x.Driver.eq('A')])
    assert not self_only.close.any()
    nonpit=x.copy(); nonpit.loc[nonpit.Driver.eq('B'),'PitInTime']=pd.Timedelta(seconds=98.)
    got=start_exposure(nonpit)
    assert not got.loc[got.Driver.eq('A')&got.LapNumber.eq(3),'close'].iloc[0]


def test_pairing():
    x=pd.DataFrame(dict(Driver=['A']*6,LapNumber=[2,3,4,5,6,7],Stint=[1]*6,Compound=['HARD']*6,
        TyreLife=[2.,3.,4.,5.,6.,7.],Position=[1.]*6,close=[False,True,False,True,False,True]))
    pairs=disjoint_pairs(x)
    assert list(zip(pairs.lap1,pairs.lap2))==[(2,3),(4,5),(6,7)]
    used=pairs[['lap1','lap2']].to_numpy().ravel()
    assert len(used)==len(set(used))
    for column,value in [('Stint',2),('Compound','SOFT'),('TyreLife',99.),('Position',3.)]:
        bad=x.iloc[:2].copy(); bad.loc[bad.index[1],column]=value
        assert disjoint_pairs(bad).empty
    assert disjoint_pairs(x.iloc[:0]).empty


if __name__=='__main__':
    test_start_exposure()
    test_pairing()
    print('lap-start timing, different lap counts, self/pit exclusion and disjoint controlled pairs passed')
