"""E193: original nine-column hazard features, with E192 draws and settings."""
import numpy as np

from experiments.e192_simulated_winner import add_reliability,main
from f1.profile import COLL,PU,PU_FAIL  # constants only; never load_aux or table()

FULL_DNF_F=['d10_dnf','t10_dnf','d3_dnf','t3_dnf','d_coll','c_dnf','c_coll','pu_pufail','era_round']


def supplier(season,team):
    return next((mapping.get(team) for (first,last),mapping in PU.items() if first<=season<=last),None)


def add_complete_reliability(X,results):
    out=add_reliability(X,results)
    keys=['season','round','code']
    source=results[keys+['circuit','team','status']].merge(X[keys+['k','era']],on=keys,validate='one_to_one')
    source['pu']=[supplier(s,t) for s,t in zip(source.season,source.team)]
    source['pufail']=source.status.isin(PU_FAIL).astype(float)
    source['coll']=source.status.isin(COLL).astype(float)
    circuit=source.groupby(['k','circuit']).coll.mean().reset_index()
    out['c_coll']=np.nan; out['pu_pufail']=np.nan
    for k,cur in out.groupby('k',sort=True):
        past=source[source.k.lt(k)]
        weight=.5**((k-past.k)/6)*np.where(past.era<cur.era.iloc[0],.5,1.)
        numerator=(past.pufail*weight).where(past.pufail.notna()).groupby(past.pu).sum()
        denominator=(past.pufail.notna()*weight).groupby(past.pu).sum().replace(0,np.nan)
        current_pu=[supplier(s,t) for s,t in zip(cur.season,cur.team)]
        out.loc[cur.index,'pu_pufail']=[(numerator/denominator).get(pu,np.nan) if pu is not None else np.nan for pu in current_pu]
        name=source[source.k.eq(k)].circuit.iloc[0]
        out.loc[cur.index,'c_coll']=circuit[circuit.k.lt(k)&circuit.circuit.eq(name)].coll.mean()
    return out


if __name__=='__main__':
    from f1.data import load_results
    source=load_results().loc[lambda d:d.season.le(2022)]
    missing=source[[supplier(s,t) is None for s,t in zip(source.season,source.team)]]
    print('original PU map unmapped season/team:',missing.groupby(['season','team']).size().to_dict(),flush=True)
    main(reliability_builder=add_complete_reliability,reliability_features=FULL_DNF_F,experiment=193)
