"""E188: fixed pre-2019 scales for historical team fusion, inner-only winner audit."""
import numpy as np
import pandas as pd

import f1.walkforward as W
from f1.data import DATA, load_results

INDICATORS=['pace_gap','fcorr_gap','q1gap','fp_lr','ptsneg','order']


def centered_team_table(results, pace, fuel, sessions):
    keys=['season','round','code']
    d=results.copy()
    d['q1gap']=(d.q1/d.groupby(['season','round']).q1.transform('min')-1)*100
    d['ptsneg']=-d.points
    fp=sessions[sessions.session.isin(['FP1','FP2','FP3'])].groupby(keys).lr_gap.mean().rename('fp_lr').reset_index()
    for table in (pace[keys+['pace_gap']],fuel[keys+['fcorr_gap']],fp):
        d=d.merge(table,on=keys,how='left',validate='one_to_one')
    t=d.groupby(['season','round','team'],sort=True)[INDICATORS].mean().reset_index()
    t[INDICATORS]=t[INDICATORS]-t.groupby(['season','round'])[INDICATORS].transform('mean')
    return t


def freeze_scales(table, columns=INDICATORS):
    scale=table[table.season.eq(2018)][columns].std()
    assert np.isfinite(scale).all() and scale.gt(0).all(), 'missing or zero 2018 calibration'
    return scale


def filter_teams(table, columns, variance, shuffled=False):
    assert table.season.le(2022).all(), 'inner only'
    assert not table.duplicated(['season','round','team']).any()
    scale=freeze_scales(table)
    d=table.copy(); d[INDICATORS]=d[INDICATORS]/scale
    state={}; previous=None; rows=[]; rng=np.random.default_rng(188)
    for (season,round_),g in d.groupby(['season','round'],sort=True):
        g=g.sort_values('team').reset_index(drop=True)
        if shuffled:
            observed=g[INDICATORS].to_numpy()[rng.permutation(len(g))]
        else:
            observed=g[INDICATORS].to_numpy()
        for i,team in enumerate(g.team):
            m,p=state.get(team,(0.,3.))
            p+=.03+(.4 if previous is not None and season!=previous else 0.)
            if season>=2019:
                rows.append(dict(season=season,round=round_,team=team,strength=m))
            for column in columns:
                value=observed[i,INDICATORS.index(column)]
                if np.isfinite(value):
                    gain=p/(p+variance)
                    m+=gain*(value-m); p*=1-gain
            state[team]=(m,p)
        previous=season
    return pd.DataFrame(rows)


def source_table():
    results=load_results().loc[lambda d:d.season.le(2022)]
    pace=pd.read_csv(DATA/'racepace_ff1.csv').loc[lambda d:d.season.le(2022)]
    fuel=pd.concat([pd.read_csv(DATA/'parts'/name) for name in ('pacev_2018_2021.csv','pacev_2022_2024.csv')],ignore_index=True).loc[lambda d:d.season.le(2022)]
    sessions=pd.concat([pd.read_csv(DATA/'parts'/name) for name in ('sessions_2018_2019.csv','sessions_2020_2022.csv')],ignore_index=True)
    sessions=sessions[sessions.season.le(2022)]
    return centered_team_table(results,pace,fuel,sessions),results


def promote_team(g,strength):
    chosen_team=strength.sort_values(['strength','team']).iloc[0].team
    ranked=g.sort_values(['base_score','driver'])
    selected=ranked[ranked.team.eq(chosen_team)].iloc[0].driver
    scores=pd.Series(np.arange(len(ranked),dtype=float),index=ranked.driver).reindex(g.driver).to_numpy()
    scores[g.driver.eq(selected).to_numpy()]=-1.
    return scores


def main():
    table,results=source_table()
    forecasts={
        'fusion_equal':filter_teams(table,INDICATORS,2.),
        'fusion_single':filter_teams(table,['pace_gap'],1.),
        'fusion_null':filter_teams(table,INDICATORS,2.,shuffled=True),
    }
    base=pd.read_csv('results/e181_inner_predictions.csv',float_precision='round_trip')
    winner=base[base.model.eq('win_logit')].rename(columns={'score':'base_score'})
    d=results[results.season.ge(2020)].merge(winner[['season','round','driver','base_score']],on=['season','round','driver'],validate='one_to_one')
    rows=[]
    for (season,round_),g in d.groupby(['season','round'],sort=True):
        scores={}
        for name,f in forecasts.items():
            strength=f[f.season.eq(season)&f['round'].eq(round_)]
            assert set(strength.team)==set(g.team)
            scores[name]=promote_team(g,strength)
        for name in ('standings','ens','win_logit'):
            saved=base[base.season.eq(season)&base['round'].eq(round_)&base.model.eq(name)]
            scores[name]=saved.set_index('driver').score.reindex(g.driver).to_numpy()
        actual=g[g.order.eq(1)].iloc[0]
        for name,score in scores.items():
            chosen=g.assign(score=score).sort_values(['score','driver']).iloc[:3]
            rows.append(dict(season=season,round=round_,model=name,**W.metrics(g,score,tie=None),
                team_hit=float(chosen.iloc[0].team==actual.team),pred1=chosen.iloc[0].driver,pred2=chosen.iloc[1].driver,pred3=chosen.iloc[2].driver))
    R=pd.DataFrame(rows)
    assert R.season.max()==2022 and R.groupby('model').size().eq(61).all()
    print('2018 scales',freeze_scales(table).to_dict(),flush=True)
    print(R.groupby('model')[['winner','podium','team_hit']].agg(['sum','mean']).round(4).to_string(),flush=True)
    print(R.pivot_table(index='season',columns='model',values='winner',aggfunc='sum').to_string(),flush=True)
    pivot=R.pivot(index=['season','round'],columns='model',values='winner')
    cis={}
    for ref in ('standings','ens','win_logit','fusion_single'):
        delta,ci,p=W.paired_boot(pivot.fusion_equal,pivot[ref]); cis[ref]=(delta,ci)
        print('fusion_equal minus',ref,delta,ci,p,flush=True)
    gate=pivot.fusion_equal.sum()>=35 and R[R.model.eq('fusion_equal')].team_hit.sum()>=40 and all(cis[ref][0]>0 and cis[ref][1][0]>0 for ref in ('win_logit','fusion_single')) and pivot.fusion_equal.sum()>pivot.fusion_null.sum()
    print('INNER GATE',bool(gate),flush=True)
    R.to_csv('results/e188_inner_per_race.csv',index=False)
    pd.concat([f.assign(model=name) for name,f in forecasts.items()],ignore_index=True).to_csv('results/e188_inner_team_forecasts.csv',index=False)


if __name__=='__main__':
    main()
