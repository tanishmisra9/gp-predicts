"""E281 independent all-raw source reduction and registered stopping-rule audit."""
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import fastf1
import numpy as np
import pandas as pd

import f1.ff1
from experiments.e190_traffic_feasibility import cached_session
from experiments.e281_compound_contrast import entries, prior
from f1.data import ROOT


def main():
    result = json.loads((ROOT/'results/e281_summary.json').read_text())
    saved = pd.read_csv(ROOT/'results/e281_stints.csv',float_precision='round_trip')
    teams = pd.read_csv(ROOT/'results/e281_teams.csv',float_precision='round_trip',parse_dates=['date'])
    history = pd.read_csv(ROOT/'results/e281_history.csv',float_precision='round_trip',parse_dates=['date'])
    roster = entries(); reconstructed = []; reports = []; raw_tables = {}
    fastf1.Cache.offline_mode(True)
    with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('ALL HTTP forbidden')):
        for rnd in range(1,22):
            session = cached_session(fastf1,2019,rnd); raw = pd.DataFrame(session.laps); raw_tables[rnd] = raw
            cache = Path(fastf1.Cache._get_cache_file_path(session.api_path,'_extended_timing_data'))
            h = hashlib.sha256(cache.read_bytes()).hexdigest()
            assert not raw.duplicated(['Driver','LapNumber']).any()
            keep = raw.LapNumber.gt(1)&raw.LapTime.notna()&raw.Time.notna()&raw.LapStartTime.notna()
            keep &= raw.PitInTime.isna()&raw.PitOutTime.isna()&raw.TrackStatus.astype(str).eq('1')&raw.IsAccurate.fillna(False)
            clean = raw[keep].copy(); clean['seconds'] = clean.LapTime.dt.total_seconds()
            med = {d:float(np.median(g.seconds)) for d,g in clean.groupby('Driver')}
            clean = clean[[r.seconds<1.07*med[r.Driver] for r in clean.itertuples()]]
            assert np.isfinite(clean.seconds).all() and (clean.seconds>0).all()
            dry = clean[clean.Compound.isin(['SOFT','MEDIUM','HARD'])].copy()
            fields = {n:(len(g),float(np.median(g.seconds))) for n,g in dry.groupby('LapNumber')}
            eligible = []; observations = {}
            for r in dry.itertuples():
                if fields[r.LapNumber][0]<6 or r.Compound not in ['MEDIUM','HARD'] or not np.isfinite(r.TyreLife): continue
                if not 5<=r.TyreLife<=15 or r.TyreLife!=int(r.TyreLife): continue
                eligible.append(r)
                observations.setdefault((r.Driver,r.Stint,r.Compound),[]).append((int(r.TyreLife),100*(r.seconds/fields[r.LapNumber][1]-1)))
            count = 0
            for (driver,stint,compound), observations_ in sorted(observations.items()):
                ages = [a for a,v in observations_]; halves = [[v for a,v in observations_ if a%2==p] for p in [0,1]]
                if len(ages)<6 or len(set(ages))!=len(ages) or min(map(len,halves))<3: continue
                count += 1
                reconstructed.append(dict(driver=driver,stint=stint,compound=compound,n=len(ages),n0=len(halves[0]),n1=len(halves[1]),
                                          response=math.fsum(v for a,v in observations_)/len(ages),half0=math.fsum(halves[0])/len(halves[0]),half1=math.fsum(halves[1])/len(halves[1]),season=2019,round=rnd))
            reports.append(dict(round=rnd,timing_sha256=h,clean_dry_laps=len(dry),candidate_laps=len(eligible),qualified_stints=count))
            assert hashlib.sha256(cache.read_bytes()).hexdigest()==h
    expected = pd.DataFrame(reconstructed,columns=saved.columns); keys = ['round','driver','stint','compound']
    pd.testing.assert_frame_equal(saved.sort_values(keys).reset_index(drop=True),expected.sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-11,rtol=1e-11)
    assert result['source']==reports and result['recorded_stints']==len(expected)
    compound_means = {}
    for key,g in expected.groupby(['round','driver','compound']):
        compound_means[key] = [math.fsum(g[c])/len(g) for c in ['response','half0','half1']]
    rebuilt = []
    for (rnd,team),g in roster.groupby(['round','team'],sort=True):
        d = []
        for code in sorted(g.code):
            a,b = compound_means.get((rnd,code,'MEDIUM')),compound_means.get((rnd,code,'HARD'))
            if a is not None and b is not None: d.append([x-y for x,y in zip(a,b)])
        values = [(d[0][i]+d[1][i])/2 for i in range(3)] if len(d)==2 else [float('nan')]*3
        rebuilt.append(dict(season=2019,round=int(rnd),team=team,date=g.date.iloc[0],paired_drivers=len(d),**dict(zip(['response','half0','half1'],values))))
    for row in rebuilt:
        peers = [r for r in rebuilt if r['round']==row['round'] and all(math.isfinite(r[c]) for c in ['response','half0','half1'])]
        for c in ['response','half0','half1']:
            row['centered_'+c] = row[c]-math.fsum(r[c] for r in peers)/len(peers) if len(peers)>=3 and math.isfinite(row[c]) else float('nan')
    expected_teams = pd.DataFrame(rebuilt)
    pd.testing.assert_frame_equal(teams,expected_teams[teams.columns],check_dtype=False,atol=1e-11,rtol=1e-11)
    counts = []
    for row in rebuilt:
        cutoff = row['date']-pd.Timedelta(days=(row['date'].weekday()-3)%7)
        previous = [p for p in rebuilt if p['round']<row['round'] and p['date']<cutoff and p['team']==row['team'] and math.isfinite(p['centered_response'])]
        counts.append(len(previous))
    valid = [i for i,r in enumerate(rebuilt) if math.isfinite(r['centered_response'])]
    prior3 = [i for i in valid if counts[i]>=3]
    support = dict(rows=len(valid),races=len({rebuilt[i]['round'] for i in valid}),prior3_rows=len(prior3),prior3_races=len({rebuilt[i]['round'] for i in prior3}))
    support['gate'] = support['rows']>=40 and support['races']>=12 and support['prior3_rows']>=30 and support['prior3_races']>=10
    assert support==result['coverage']
    np.testing.assert_array_equal(counts,history.past_count)
    if not support['gate']:
        assert not result['gate'] and not result['temporal_evaluated'] and 'prediction' not in history
        assert all(k not in result for k in ['mse','paired','split','relative_gain'])
    else:
        # Independent prior arithmetic and matched prefix-stable null.
        rng = np.random.default_rng(281); shuffled = {}
        for rnd in range(1,22):
            ids = [i for i in valid if rebuilt[i]['round']==rnd]; ids.sort(key=lambda i:rebuilt[i]['team'])
            for dest,src in zip(ids,rng.permutation(ids)): shuffled[dest] = rebuilt[src]['centered_response']
        direct = []; null = []
        for i,row in enumerate(rebuilt):
            cutoff = row['date']-pd.Timedelta(days=(row['date'].weekday()-3)%7)
            ids = [j for j in valid if rebuilt[j]['round']<row['round'] and rebuilt[j]['date']<cutoff and rebuilt[j]['team']==row['team']]
            weights = [2.**((rebuilt[j]['round']-row['round'])/10) for j in ids]; denominator = 5+math.fsum(weights)
            direct.append(math.fsum(w*rebuilt[j]['centered_response'] for w,j in zip(weights,ids))/denominator)
            null.append(math.fsum(w*shuffled[j] for w,j in zip(weights,ids))/denominator)
        np.testing.assert_allclose(direct,history.prediction,atol=1e-11,rtol=1e-11)
        np.testing.assert_allclose(null,history.null_prediction,atol=1e-11,rtol=1e-11)
        # Independent cluster resampling/pooled correlation and equal-race losses.
        rounds = sorted({rebuilt[i]['round'] for i in valid})
        vectors = [np.array([[rebuilt[i]['centered_half0'],rebuilt[i]['centered_half1']] for i in valid if rebuilt[i]['round']==rnd]) for rnd in rounds]
        def corr(a):
            return float(np.corrcoef(a.T)[0,1]) if np.std(a[:,0])>0 and np.std(a[:,1])>0 else float('nan')
        split = corr(np.concatenate(vectors)); index = np.random.default_rng(0).integers(0,len(rounds),size=(5000,len(rounds)))
        draws = [corr(np.concatenate([vectors[i] for i in ids])) for ids in index]
        finite = [x for x in draws if math.isfinite(x)]
        ci = np.quantile(finite,[.025,.975]) if finite else [float('nan')]*2
        np.testing.assert_allclose(split,result['split']['correlation'],atol=1e-11)
        np.testing.assert_allclose(ci,result['split']['ci'],atol=1e-11)
        assert len(finite)==result['split']['finite_draws']
        losses = []
        for rnd in rounds:
            ids = [i for i in valid if rebuilt[i]['round']==rnd]
            losses.append([math.fsum(rebuilt[i]['centered_response']**2 for i in ids)/len(ids),
                           math.fsum((direct[i]-rebuilt[i]['centered_response'])**2 for i in ids)/len(ids),
                           math.fsum((null[i]-rebuilt[i]['centered_response'])**2 for i in ids)/len(ids)])
        losses = np.array(losses); means = losses.mean(axis=0)
        np.testing.assert_allclose(means,[result['mse'][n] for n in ['zero_loss','candidate_loss','null_loss']],atol=1e-11)
        lower = []
        for j,name in [(0,'zero_loss'),(2,'null_loss')]:
            delta = losses[:,j]-losses[:,1]; sampled = delta[index].mean(axis=1); interval = np.quantile(sampled,[.025,.975]); lower.append(interval[0])
            np.testing.assert_allclose([delta.mean(),*interval],[result['paired'][name]['gain'],*result['paired'][name]['ci']],atol=1e-11)
            assert (sampled<=0).mean()==result['paired'][name]['p_nonpositive']
        relative = 1-means[1]/means[0] if means[0]>0 else None
        np.testing.assert_allclose(relative,result['relative_gain'],atol=1e-11)
        assert result['gate']==bool(split>=.25 and ci[0]>0 and relative is not None and relative>=.02 and min(lower)>0)
    # History helper exact current/future/prefix invariance, irrespective of coverage.
    first = prior(teams); target = 15
    poisoned = teams.copy(); poisoned.loc[poisoned['round'].ge(target),'centered_response'] = 12345.
    second = prior(poisoned)
    np.testing.assert_array_equal(first.loc[first['round'].le(target),'prediction'],second.loc[second['round'].le(target),'prediction'])
    prefix = prior(teams[teams['round'].le(target)])
    np.testing.assert_array_equal(prefix.prediction,first.loc[first['round'].le(target),'prediction'])
    if len(valid):
        i = valid[0]; altered = teams.copy(); altered.loc[i,'centered_response'] += 1
        later = teams['round'].gt(teams.loc[i,'round'])&teams.team.eq(teams.loc[i,'team'])
        if later.any(): assert not np.array_equal(prior(altered).loc[later,'prediction'],first.loc[later,'prediction'])
    # Actual raw current/future session poisoning, followed by full measurement rebuild.
    # These are helper invariance checks, not temporal scoring or a forecast artifact.
    from experiments.e281_compound_contrast import stints, team_table
    baseline = prior(team_table(saved,roster)); mutated_parts = []; changed_roster = roster.copy()
    changed_roster.loc[changed_roster['round'].gt(target),'code'] += '_future'
    changed_roster.loc[changed_roster['round'].gt(target),'team'] += '_future'
    for rnd,raw in raw_tables.items():
        changed = raw.copy()
        if rnd>=target:
            changed['TrackStatus']='2'; changed['LapTime']=changed.LapTime*1.001
            changed['TyreLife']=changed.TyreLife+2; changed['order']=-999; changed['grid']=999
        if rnd>target: changed['Driver']=changed.Driver+'_future'
        observed,_ = stints(changed); mutated_parts.append(observed.assign(season=2019,round=rnd))
    mutated = prior(team_table(pd.concat(mutated_parts,ignore_index=True),changed_roster))
    np.testing.assert_array_equal(baseline.loc[baseline['round'].le(target),'prediction'],mutated.loc[mutated['round'].le(target),'prediction'])
    short = prior(team_table(saved[saved['round'].le(target)],roster[roster['round'].le(target)]))
    np.testing.assert_array_equal(short.prediction,baseline.loc[baseline['round'].le(target),'prediction'])
    if len(valid):
        original = rebuilt[valid[0]]; rnd = original['round']; codes = set(roster[roster['round'].eq(rnd)&roster.team.eq(original['team'])].code)
        changed = raw_tables[rnd].copy(); selected = changed.Driver.isin(codes)&changed.Compound.eq('MEDIUM')
        changed.loc[selected,'LapTime'] += pd.Timedelta(seconds=.05)
        observed,_ = stints(changed)
        replaced = pd.concat([saved[saved['round'].ne(rnd)],observed.assign(season=2019,round=rnd)],ignore_index=True)
        after = prior(team_table(replaced,roster)); later = baseline['round'].gt(rnd)&baseline.team.eq(original['team'])
        if later.any(): assert not np.array_equal(after.loc[later,'prediction'],baseline.loc[later,'prediction'])
    assert len(teams)==result['official_team_events']==210
    assert not any(result[k] for k in ['winner_model_fitted','outer_evaluated','evaluated_2026'])
    assert result['roster_sha256']==hashlib.sha256(roster.to_csv(index=False).encode()).hexdigest()
    for n,h in result['code_hashes'].items(): assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h
    for n,h in result['artifact_hashes'].items(): assert hashlib.sha256((ROOT/'results'/n).read_bytes()).hexdigest()==h
    print('E281 independent ALL21 raw laps/filter/field/stint/parity/driver/team/210 history-count/cache/hash/stopping-rule audit PASS')


if __name__=='__main__': main()
