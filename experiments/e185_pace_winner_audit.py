"""E185: inner-only winner alignment and transfer audit; hindsight is labelled."""
import numpy as np
import pandas as pd

import f1.walkforward as W
from f1.data import DATA, load_results


MEASURES = ['med_gap','p25_gap','fcorr_gap','stint1_gap','best_stint_gap','tcorr_gap','matched_gap','joint_gap','fresh_gap']
KEYS = ['season','round','code']


def source_table():
    variants = pd.concat([pd.read_csv(DATA / 'parts' / p) for p in ('pacev_2018_2021.csv','pacev_2022_2024.csv')], ignore_index=True)
    out = variants[variants.season.le(2022)][KEYS + MEASURES[:6]]
    for file, column, name in (('matchedpace_2018_2025.csv','matched_gap','matched_gap'),
                                ('jointpace_2018_2025.csv','joint_gap','joint_gap'),
                                ('jointpace_fresh_2018_2025.csv','joint_gap','fresh_gap')):
        table = pd.read_csv(DATA / 'parts' / file).loc[lambda d:d.season.le(2022),KEYS + [column]]
        out = out.merge(table.rename(columns={column:name}), on=KEYS, how='outer', validate='one_to_one')
    assert not out.duplicated(KEYS).any() and out.season.le(2022).all()
    return out.sort_values(KEYS).reset_index(drop=True)


def shuffled_source(source):
    out = source.copy()
    rng = np.random.default_rng(185)
    for _, ids in out.groupby(['season','round'],sort=True).groups.items():
        # One assignment permutation for all columns preserves their joint structure.
        out.loc[ids,MEASURES] = out.loc[rng.permutation(ids),MEASURES].to_numpy()
    return out


def prior_measurements(entries, source):
    events = entries[['season','round','k']].drop_duplicates()
    history = source.merge(events,on=['season','round'],validate='many_to_one')
    out = entries[['season','round','k','driver','code']].copy()
    out[MEASURES] = np.nan
    for k,cur in entries.groupby('k',sort=True):
        past = history[history.k.lt(k)]
        w = pd.Series(.5 ** ((k-past.k)/10),index=past.index)
        for col in MEASURES:
            valid = past[col].notna()
            rate = (past.loc[valid,col]*w[valid]).groupby(past.loc[valid,'code']).sum() / w[valid].groupby(past.loc[valid,'code']).sum()
            out.loc[cur.index,col] = cur.code.map(rate)
    return out


def main():
    df = load_results().loc[lambda d:d.season.le(2022)].sort_values(['season','round','driver']).reset_index(drop=True)
    df['k'] = df.groupby(['season','round'],sort=True).ngroup()
    entries = df[['season','round','k','driver','code']]
    source = source_table()
    prior = prior_measurements(entries,source)
    null = prior_measurements(entries,shuffled_source(source))
    current = entries.merge(source,on=KEYS,how='left',validate='one_to_one')
    W.assert_pre_weekend(['prior_' + c for c in MEASURES])
    baseline = pd.read_csv('results/e181_inner_predictions.csv',float_precision='round_trip')
    rows = []
    for (season,round_),g in df[df.season.ge(2020)].groupby(['season','round'],sort=True):
        scores = {}
        for mode,table in (('HINDSIGHT',current),('PRIOR',prior),('NULL',null)):
            values = table.set_index(['season','round','driver']).loc[(season,round_)].reindex(g.driver)
            for col in MEASURES:
                scores[mode + '_' + col] = values[col].fillna(np.inf).to_numpy()
        for name in ('standings','ens','win_logit'):
            saved = baseline[baseline.season.eq(season) & baseline['round'].eq(round_) & baseline.model.eq(name)]
            scores[name] = saved.set_index('driver').score.reindex(g.driver).to_numpy()
        actual = g.loc[g.order.eq(1)].iloc[0]
        for name,score in scores.items():
            metrics = W.metrics(g,score,tie=None)
            chosen = g.assign(score=score).sort_values(['score','driver']).iloc[:3]
            assert metrics['winner'] == float(chosen.iloc[0].driver == actual.driver)
            assert metrics['podium'] == len(set(chosen.driver) & set(g[g.order.le(3)].driver))/3
            row = dict(season=season,round=round_,model=name,**metrics,
                       team_hit=float(chosen.iloc[0].team == actual.team),
                       pred1=chosen.iloc[0].driver,pred2=chosen.iloc[1].driver,pred3=chosen.iloc[2].driver)
            if name.startswith('HINDSIGHT_'):
                col = name.removeprefix('HINDSIGHT_')
                v = current[current.season.eq(season) & current['round'].eq(round_) & current.driver.eq(actual.driver)][col]
                row['winner_source_available'] = bool(v.notna().iloc[0])
            rows.append(row)
    R = pd.DataFrame(rows)
    assert R.groupby('model').size().eq(61).all() and R.season.max() == 2022
    print(R.groupby('model')[['winner','podium','team_hit']].mean().round(4).to_string(),flush=True)
    print('hindsight winner coverage',R[R.model.str.startswith('HINDSIGHT')].groupby('model').winner_source_available.sum().to_dict(),flush=True)
    by = R.pivot(index=['season','round'],columns='model',values='winner')
    for col in MEASURES:
        candidate = by['PRIOR_'+col]
        for ref in ('standings','ens','win_logit','PRIOR_med_gap'):
            print('PRIOR_'+col,'minus',ref,W.paired_boot(candidate,by[ref]),flush=True)
        diff,ci,_ = W.paired_boot(candidate,by.PRIOR_med_gap)
        gate = by['HINDSIGHT_'+col].sum()>=46 and candidate.sum()>=35 and diff>0 and ci[0]>0 and candidate.sum()>by['NULL_'+col].sum()
        print(col,'GATE',bool(gate),flush=True)
    R.to_csv('results/e185_inner_measurement_audit.csv',index=False)


if __name__ == '__main__':
    main()
