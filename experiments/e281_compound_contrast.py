"""E281 fixed observational team MEDIUM-minus-HARD contrast, cached 2019 only."""
import csv
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.e190_traffic_feasibility import cached_session, start_exposure
from f1.data import DATA, ROOT

COLS = ['driver', 'stint', 'compound', 'n', 'n0', 'n1', 'response', 'half0', 'half1']
VALUES = ['response', 'half0', 'half1']


def entries():
    rows = []
    with (DATA/'results.csv').open() as file:
        for r in csv.DictReader(file):
            if r['season'] == '2019':
                rows.append(dict(season=2019, round=int(r['round']), code=r['code'], team=r['team'], date=r['date']))
    result = pd.DataFrame(rows); result['date'] = pd.to_datetime(result.date)
    assert len(result) == 420 and result['round'].nunique() == 21
    assert not result.duplicated(['round','code']).any() and result.groupby(['round','team']).size().eq(2).all()
    assert result.groupby('round').date.nunique().eq(1).all() and result.date.notna().all()
    return result


def stints(laps):
    assert not laps.duplicated(['Driver','LapNumber']).any()
    clean = start_exposure(laps)
    assert np.isfinite(clean.seconds).all() and clean.seconds.gt(0).all()
    clean = clean[clean.Compound.isin(['SOFT','MEDIUM','HARD'])].copy()
    field = clean.groupby('LapNumber').seconds
    clean['relative'] = 100*(clean.seconds/field.transform('median')-1)
    eligible = clean[field.transform('count').ge(6) & clean.Compound.isin(['MEDIUM','HARD'])
                     & np.isfinite(clean.TyreLife) & clean.TyreLife.between(5,15)
                     & clean.TyreLife.eq(np.floor(clean.TyreLife))]
    rows = []
    for (driver, stint, compound), g in eligible.groupby(['Driver','Stint','Compound'], sort=True):
        half = [g[g.TyreLife.mod(2).eq(h)] for h in [0,1]]
        if len(g)<6 or g.TyreLife.nunique()!=len(g) or any(len(h)<3 for h in half): continue
        rows.append(dict(driver=driver,stint=stint,compound=compound,n=len(g),n0=len(half[0]),n1=len(half[1]),
                         response=float(g.relative.mean()),half0=float(half[0].relative.mean()),half1=float(half[1].relative.mean())))
    return pd.DataFrame(rows,columns=COLS), dict(clean_dry_laps=len(clean),candidate_laps=len(eligible),qualified_stints=len(rows))


def team_table(source, roster):
    assert roster.season.eq(2019).all() and source.season.eq(2019).all()
    assert not roster.duplicated(['round','code']).any() and roster.groupby(['round','team']).size().eq(2).all()
    assert not source.duplicated(['round','driver','stint','compound']).any()
    assert np.isfinite(source[VALUES].to_numpy(float)).all()
    legal = set(roster[['round','code']].itertuples(index=False,name=None))
    assert set(source[['round','driver']].itertuples(index=False,name=None)) <= legal
    assert source.compound.isin(['MEDIUM','HARD']).all()
    means = source.groupby(['round','driver','compound'])[VALUES].mean()
    rows = []
    for (rnd,team), group in roster.groupby(['round','team'],sort=True):
        assert group.date.nunique()==1
        contrasts = []
        for driver in sorted(group.code):
            if (rnd,driver,'MEDIUM') in means.index and (rnd,driver,'HARD') in means.index:
                contrasts.append((means.loc[(rnd,driver,'MEDIUM')]-means.loc[(rnd,driver,'HARD')]).to_numpy())
        values = np.mean(contrasts,axis=0) if len(contrasts)==2 else [np.nan]*3
        rows.append(dict(season=2019,round=int(rnd),team=team,date=group.date.iloc[0],paired_drivers=len(contrasts),**dict(zip(VALUES,values))))
    table = pd.DataFrame(rows)
    common = table[VALUES].notna().all(axis=1)
    for col in VALUES:
        group = table[col].where(common).groupby(table['round'])
        table['centered_'+col] = (table[col]-group.transform('mean')).where(common & group.transform('count').ge(3))
    return table


def prior(table, predict=True):
    assert table.season.eq(2019).all() and table.date.notna().all()
    assert not table.duplicated(['round','team']).any() and table.groupby('round').date.nunique().eq(1).all()
    assert table[['round','date']].drop_duplicates().sort_values('round').date.is_monotonic_increasing
    assert np.isfinite(table.centered_response.dropna().to_numpy(float)).all()
    out = table.copy(); out['past_count'] = 0
    if predict: out['prediction'] = 0.
    for i,r in out.iterrows():
        cutoff = r.date.normalize()-pd.Timedelta(days=(r.date.weekday()-3)%7)
        past = table[table['round'].lt(r['round']) & table.date.lt(cutoff) & table.team.eq(r.team) & table.centered_response.notna()]
        out.loc[i,'past_count'] = len(past)
        if predict:
            weight = 2.**((past['round']-r['round'])/10)
            out.loc[i,'prediction'] = (past.centered_response*weight).sum()/(5+weight.sum())
    return out


def coverage(table):
    support = prior(table,predict=False); valid = support.centered_response.notna()
    old = valid & support.past_count.ge(3)
    result = dict(rows=int(valid.sum()),races=int(support.loc[valid,'round'].nunique()),
                  prior3_rows=int(old.sum()),prior3_races=int(support.loc[old,'round'].nunique()))
    result['gate'] = result['rows']>=40 and result['races']>=12 and result['prior3_rows']>=30 and result['prior3_races']>=10
    return support,result


def split_interval(table):
    a = table.dropna(subset=['centered_half0','centered_half1']); moments = []
    for _,g in a.groupby('round',sort=True):
        x,y = g.centered_half0.to_numpy(),g.centered_half1.to_numpy()
        moments.append([len(g),x.sum(),y.sum(),x@x,y@y,x@y])
    moments = np.array(moments)
    def correlation(s):
        n,x,y,xx,yy,xy = s.T
        denominator = (xx-x*x/n)*(yy-y*y/n)
        return np.divide(xy-x*y/n,np.sqrt(np.maximum(denominator,0)),out=np.full_like(n,np.nan,dtype=float),where=denominator>0)
    point = float(correlation(moments.sum(axis=0)))
    indices = np.random.default_rng(0).integers(0,len(moments),size=(5000,len(moments)))
    draws = correlation(moments[indices].sum(axis=1)); finite = draws[np.isfinite(draws)]
    return dict(correlation=point,ci=np.quantile(finite,[.025,.975]).tolist() if len(finite) else [None,None],finite_draws=len(finite))


def evaluate(table):
    support,c = coverage(table)
    result = dict(coverage=c,gate=False,temporal_evaluated=False)
    if not c['gate']: return result,support
    split = split_interval(table); predicted = prior(table); null = table.copy(); rng = np.random.default_rng(281)
    for _,g in table[table.centered_response.notna()].sort_values(['round','team']).groupby('round',sort=True):
        ids = g.index.to_numpy(); null.loc[ids,'centered_response'] = table.loc[rng.permutation(ids),'centered_response'].to_numpy()
    predicted['null_prediction'] = prior(null).prediction
    measured = predicted[predicted.centered_response.notna()].copy()
    measured['zero_loss'] = measured.centered_response**2
    measured['candidate_loss'] = (measured.prediction-measured.centered_response)**2
    measured['null_loss'] = (measured.null_prediction-measured.centered_response)**2
    race = measured.groupby('round')[['zero_loss','candidate_loss','null_loss']].mean()
    paired = {}
    for name in ['zero_loss','null_loss']:
        diff,ci,p = W.paired_boot(race[name],race.candidate_loss,n=5000,seed=0)
        paired[name] = dict(gain=float(diff),ci=ci.tolist(),p_nonpositive=float(p))
    means = race.mean().to_dict(); relative = 1-means['candidate_loss']/means['zero_loss'] if means['zero_loss']>0 else None
    result.update(split=split,temporal_evaluated=True,mse=means,relative_gain=relative,paired=paired)
    result['gate'] = bool(split['correlation']>=.25 and split['ci'][0] is not None and split['ci'][0]>0
                          and relative is not None and relative>=.02 and all(p['ci'][0]>0 for p in paired.values()))
    return result,predicted


def main():
    import fastf1
    import f1.ff1
    from fastf1.internals import f1account
    assert not (ROOT/'results/e281_summary.json').exists(), 'do not overwrite completed measurement'
    roster = entries(); parts = []; reports = []; fastf1.Cache.offline_mode(True)
    with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('E281 ALL HTTP forbidden')), patch.object(f1account,'get_token',side_effect=RuntimeError('E281 token lookup forbidden')):
        for rnd in range(1,22):
            session = fastf1.get_session(2019,rnd,'R')
            cache = Path(fastf1.Cache._get_cache_file_path(session.api_path,'_extended_timing_data'))
            before = hashlib.sha256(cache.read_bytes()).hexdigest()
            session = cached_session(fastf1,2019,rnd)
            source,counts = stints(pd.DataFrame(session.laps)); parts.append(source.assign(season=2019,round=rnd))
            assert hashlib.sha256(cache.read_bytes()).hexdigest()==before
            reports.append(dict(round=rnd,timing_sha256=before,**counts)); print('SOURCE',rnd,counts,flush=True)
    source = pd.concat(parts,ignore_index=True); table = team_table(source,roster)
    assert len(table)==210
    result,forecast = evaluate(table)
    for name,frame in [('stints',source),('teams',table),('history',forecast)]: frame.to_csv(ROOT/f'results/e281_{name}.csv',index=False)
    result.update(source=reports,official_team_events=210,recorded_stints=len(source),winner_model_fitted=False,outer_evaluated=False,evaluated_2026=False,
                  code_hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['experiments/e281_compound_contrast.py','experiments/e190_traffic_feasibility.py','f1/walkforward.py']},
                  roster_sha256=hashlib.sha256(roster.to_csv(index=False).encode()).hexdigest())
    result['artifact_hashes'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'results').glob('e281_*.csv')}
    (ROOT/'results/e281_summary.json').write_text(json.dumps(result,indent=2)+'\n'); print('RESULT',json.dumps(result),flush=True)


if __name__=='__main__': main()
