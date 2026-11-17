"""E211 speed crossings require complete observed monotone brake-on runs."""
import numpy as np
import pandas as pd

from experiments.e211_braking_signature import braking,team_table


def test_segments():
    origin=pd.Timestamp('2019-01-01')
    t=np.arange(6)*.25
    car=pd.DataFrame(dict(Date=origin+pd.to_timedelta(t,unit='s'),Time=pd.to_timedelta(t,unit='s'),Speed=[280,260,240,210,170,140],Brake=True))
    out=braking(car,origin)
    assert len(out)==1 and out.samples.iloc[0]==5
    np.testing.assert_allclose(out.start.iloc[0],.375)
    np.testing.assert_allclose(out.end.iloc[0],1+1/6)
    np.testing.assert_allclose(out.duration.iloc[0],1+1/6-.375)
    bad=car.copy(); bad.loc[3,'Time']=pd.NaT
    assert braking(bad,origin).empty
    bad=car.copy(); bad.loc[3,'Brake']=False
    assert braking(bad,origin).empty
    bad=car.copy(); bad.loc[3:,'Date']+=pd.Timedelta(seconds=.4)
    assert braking(bad,origin).empty
    bad=car.copy(); bad.loc[3,'Speed']=245
    assert braking(bad,origin).empty
    assert braking(car.iloc[:0],origin).empty and braking(car.iloc[:3],origin).empty
    pd.testing.assert_frame_equal(braking(car.assign(DRS=999),origin),out)


def test_aggregation():
    entries=pd.DataFrame([dict(season=2019,round=r,team=f't{j}',code=d)
        for r in range(1,7) for j in range(6) for d in (f'a{j}',f'b{j}')])
    source=pd.DataFrame([dict(season=2019,round=r,driver=d,lap=lap,first_sample=lap*100,duration=1+j*.05)
        for r in range(1,7) for j in range(6) for d in (f'a{j}',f'b{j}') for lap in range(2,12)])
    table=team_table(source,entries)
    assert table.centered_response.notna().all() and table.centered_half0.notna().all() and table.centered_half1.notna().all()
    np.testing.assert_allclose(table[table.team.eq('t0')].centered_response,-.125)
    extra=source.iloc[:1].copy(); extra.first_sample+=1
    other=extra.copy(); other.first_sample+=1; other.duration=3
    pd.testing.assert_frame_equal(team_table(pd.concat([source,extra,other]),entries),table)
    assert team_table(source[source.lap.ne(2)],entries).response.isna().all()
    try:
        team_table(source.assign(season=2026),entries)
    except AssertionError:
        pass
    else:
        raise AssertionError('non2019 source accepted')


if __name__=='__main__':
    test_segments(); test_aggregation()
    print('braking crossings, padding/brake/gap/monotonicity/DRS independence and lap weighting PASS')
