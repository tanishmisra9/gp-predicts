"""E281 hand arithmetic, support boundaries, identities and strict history."""
import numpy as np
import pandas as pd

from experiments.e281_compound_contrast import coverage, evaluate, prior, split_interval, stints, team_table


def laps():
    rows = []
    for d in range(6):
        for stint, compound in [(1,'MEDIUM'),(2,'HARD')]:
            for age in range(5,11):
                n = 2+(stint-1)*6+age-5
                seconds = 100.+(d//2-1 if compound=='MEDIUM' else 0)
                rows.append(dict(Driver=f'd{d}',LapNumber=n,Stint=stint,Compound=compound,TyreLife=float(age),
                                 LapTime=pd.Timedelta(seconds=seconds),Time=pd.Timedelta(seconds=110*n+d/10),
                                 LapStartTime=pd.Timedelta(seconds=110*(n-1)+d/10),PitInTime=pd.NaT,PitOutTime=pd.NaT,
                                 TrackStatus='1',IsAccurate=True))
    return pd.DataFrame(rows)


def main():
    raw = laps(); measured,counts = stints(raw)
    assert len(measured)==12 and counts==dict(clean_dry_laps=72,candidate_laps=72,qualified_stints=12)
    for r in measured.itertuples():
        expected = int(r.driver[1:])//2-1 if r.compound=='MEDIUM' else 0.
        np.testing.assert_allclose([r.response,r.half0,r.half1],expected,atol=1e-13)
        assert (r.n,r.n0,r.n1)==(6,3,3)
    bad = raw.drop(raw[(raw.Driver=='d0') & raw.Stint.eq(1)].index[:1])
    assert not ((stints(bad)[0].driver=='d0') & stints(bad)[0].compound.eq('MEDIUM')).any()
    edge = raw.copy(); edge.loc[edge.TyreLife.eq(9),'TyreLife']=15
    assert len(stints(edge)[0])==12  # 5 and 15 inclusive, still three odd/even ages.
    out = edge.copy(); out.loc[out.TyreLife.eq(15),'TyreLife']=17
    assert len(stints(out)[0])==0
    parity = raw.copy(); parity['TyreLife'] = parity.TyreLife.map(dict(zip(range(5,11),[5,6,8,10,12,14])))
    assert stints(parity)[0].empty  # Six distinct ages but insufficient odd ages.
    duplicate_age = raw.copy(); duplicate_age.loc[duplicate_age.TyreLife.eq(10),'TyreLife']=8
    assert stints(duplicate_age)[0].empty
    assert stints(raw.iloc[:0])[0].columns.tolist()==measured.columns.tolist()
    roster = pd.DataFrame([dict(season=2019,round=r,team=f't{d//2}',code=f'd{d}',date=pd.Timestamp('2019-01-06')+pd.Timedelta(days=7*(r-1))) for r in range(1,7) for d in range(6)])
    source = pd.concat([measured.assign(season=2019,round=r) for r in range(1,7)],ignore_index=True)
    table = team_table(source,roster)
    np.testing.assert_allclose(table[table.team.eq('t0')].centered_response,-1,atol=1e-13)
    extra = source[source.driver.eq('d0') & source.compound.eq('MEDIUM')].copy().assign(stint=3)
    extra[['response','half0','half1']] = -3.
    weighted = team_table(pd.concat([source,extra],ignore_index=True),roster)
    # Equal stint mean gives driver d0=-2, teammate d1=-1, team=-1.5 before field centering.
    np.testing.assert_allclose(weighted[weighted.team.eq('t0')].response,-1.5,atol=1e-13)
    missing = source[~source.driver.eq('d0')]
    assert team_table(missing,roster).centered_response.isna().all()  # Only two complete teams.
    original = prior(table); target = 4
    changed = table.copy(); changed.loc[changed['round'].ge(target),['centered_response','centered_half0','centered_half1']]=999.
    again = prior(changed)
    np.testing.assert_array_equal(original.loc[original['round'].le(target),'prediction'],again.loc[again['round'].le(target),'prediction'])
    prefix = prior(table[table['round'].le(target)])
    np.testing.assert_array_equal(prefix.prediction,original.loc[original['round'].le(target),'prediction'])
    changed = table.copy(); changed.loc[changed['round'].eq(1)&changed.team.eq('t0'),'centered_response'] += 2
    assert prior(changed).loc[original['round'].eq(4)&original.team.eq('t0'),'prediction'].iloc[0] != original.loc[original['round'].eq(4)&original.team.eq('t0'),'prediction'].iloc[0]
    # A preceding event with date exactly Thursday is not eligible.
    boundary = table[table['round'].le(2)].copy()
    boundary.loc[boundary['round'].eq(1),'date'] = boundary.loc[boundary['round'].eq(2),'date'].iloc[0]-pd.Timedelta(days=3)
    assert prior(boundary).past_count.eq(0).all()
    report,state = evaluate(table)
    assert not report['coverage']['gate'] and not report['temporal_evaluated'] and 'prediction' not in state
    np.testing.assert_allclose(split_interval(table)['correlation'],1,atol=1e-14)
    assert split_interval(table.assign(centered_half0=0))['finite_draws']==0
    long = pd.concat([table[table['round'].eq(1)].assign(round=r,date=pd.Timestamp('2019-01-06')+pd.Timedelta(days=7*(r-1))) for r in range(1,21)],ignore_index=True)
    admitted,forecasts = evaluate(long)
    assert admitted['coverage']['gate'] and admitted['temporal_evaluated'] and admitted['gate']
    assert len(forecasts)==60 and 'null_prediction' in forecasts
    for bad in [pd.concat([source,source.iloc[:1]]),source.assign(season=2026),source.assign(driver='UNKNOWN'),source.assign(response=float('inf'))]:
        try: team_table(bad,roster)
        except AssertionError: pass
        else: raise AssertionError('bad source accepted')
    try: stints(pd.concat([raw,raw.iloc[:1]]))
    except AssertionError: pass
    else: raise AssertionError('duplicate lap accepted')
    print('E281 hand contrasts/parity/age/count/stint weighting/2driver/3team/strict prior/prefix/response/coverage-stop/invalid-source tests PASS')


if __name__=='__main__': main()
