"""E191 drift balancing, disjoint halves and strict historical transfer checks."""
import numpy as np
import pandas as pd
from experiments.e191_traffic_repeatability import balanced_response,past_response,team_measurements


def test_balancing():
    p=pd.DataFrame(dict(entering=[True]*6+[False]*2,contrast=[.4]*6+[.2]*2))
    assert np.isclose(balanced_response(p),.3)
    drift=p.copy(); drift.contrast+=np.where(p.entering,.1,-.1)
    assert np.isclose(balanced_response(drift),.3)
    assert pd.isna(balanced_response(p.iloc[:-1]))
    # Different normalization denominators prevent exact absolute-second cancellation.
    assert not np.isclose(.5*(100/90-100/100),0.)


def test_temporal():
    table=pd.DataFrame(dict(season=[2019]*10,round=[1,1,2,2,3,3,4,4,5,5],team=['a','b']*5,
        centered_response=[1.,-1.,2.,-2.,3.,-3.,4.,-4.,5.,-5.]))
    before=past_response(table)
    assert before.loc[before['round'].eq(1),'prediction'].eq(0).all()
    expected=.5**(.1)/(5+.5**(.1))
    assert np.isclose(before.loc[before['round'].eq(2)&before.team.eq('a'),'prediction'].iloc[0],expected)
    bad=table.copy(); bad.loc[bad['round'].ge(4),'centered_response']=999.
    after=past_response(bad)
    pd.testing.assert_frame_equal(before.loc[before['round'].le(4),['prediction','past_count']],after.loc[after['round'].le(4),['prediction','past_count']],check_exact=True)
    assert not before.loc[before['round'].eq(5),'prediction'].equals(after.loc[after['round'].eq(5),'prediction'])


def test_halves_and_coverage():
    rows=[]
    for i in range(6):
        for j in range(8):
            rows.append(dict(season=2019,round=1,team=str(i),driver=str(i),lap1=2*j,completion2=j,
                entering=j%4<2,contrast=i+.1*j))
    p=pd.DataFrame(rows)
    entries=p[['season','round','team']].drop_duplicates()
    out=team_measurements(p,entries)
    assert out[['response','half0','half1','centered_response','centered_half0','centered_half1']].notna().all().all()
    np.testing.assert_allclose(out[['centered_response','centered_half0','centered_half1']].mean(),0.,atol=1e-14)
    object_pairs=p.copy(); object_pairs['entering']=object_pairs.entering.astype(object)
    pd.testing.assert_frame_equal(out,team_measurements(object_pairs,entries),check_exact=True)
    missing=team_measurements(p[p.team.ne('5')],entries)
    assert missing.centered_response.isna().all()


if __name__=='__main__':
    test_balancing(); test_temporal(); test_halves_and_coverage()
    print('traffic balancing limitation, minimum counts, centering and temporal source cutoff passed')
