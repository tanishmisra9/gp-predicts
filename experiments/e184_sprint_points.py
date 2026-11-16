"""E184: include only completed earlier sprints in the season-points prior."""
import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.e168_winner import win_score
from f1.data import DATA, load_results
from f1.features import build_features, pre_weekend_cols


POINT_FEATURES = ['prior_full_d_points', 'prior_full_t_points']


def add_full_points(X, results):
    keys = ['season','round','driver']
    assert not results.duplicated(keys).any()
    source = results[keys + ['team','points','s_points']].merge(
        X[keys + ['k']], on=keys, validate='one_to_one')
    source['total'] = source.points + source.s_points.fillna(0)
    out = X.copy()
    out[POINT_FEATURES] = np.nan
    for k, cur in X.groupby('k', sort=True):
        past = source[source.k.lt(k) & source.season.eq(cur.season.iloc[0])]
        if past.empty:
            continue
        for key, col in zip(('driver','team'), POINT_FEATURES):
            totals = past.groupby(key).total.sum()
            out.loc[cur.index, col] = cur[key].map(totals) / max(totals.max(), 1e-9)
    return out


def main():
    df = load_results().loc[lambda d: d.season.le(2022)]
    pace = pd.read_csv(DATA / 'racepace_ff1.csv').loc[lambda d: d.season.le(2022)]
    X = build_features(df, pace); base = pre_weekend_cols(X)
    X = add_full_points(X, df)
    replacement = dict(zip(('d_season_pts','t_season_pts'), POINT_FEATURES))
    candidate = [replacement.get(c,c) for c in base]
    W.assert_pre_weekend(base + candidate)
    W.MODELS.update(
        standings=lambda tr, te, F: (-te.d_season_pts.fillna(0)*100 + te.d10_order.fillna(11)*1e-3).to_numpy(),
        full_points=lambda tr, te, F: (-te.prior_full_d_points.fillna(0)*100 + te.d10_order.fillna(11)*1e-3).to_numpy(),
        pre_logit=lambda tr, te, F: W.m_logit_pod(tr, te, base, C=.03, hl=100),
        win_logit=lambda tr, te, F: win_score(tr, te, base),
        full_win=lambda tr, te, F: win_score(tr, te, candidate),
        full_null=lambda tr, te, F: win_score(tr, te, candidate, shuffled=True),
    )
    R, P = W.walk_forward(X, [2020,2021,2022], ['standings','full_points','pre_logit','win_logit','full_win','full_null'],
                          F=base + candidate, ens={'ens':['standings','pre_logit']}, tie=None)
    joined = P.merge(df[['season','round','driver','team','order']], on=['season','round','driver'], validate='many_to_one')
    truth = df[df.order.eq(1)][['season','round','team']].rename(columns={'team':'winner_team'})
    picked = joined.sort_values(['season','round','model','score','driver']).drop_duplicates(['season','round','model'])
    picked = picked.merge(truth, on=['season','round'], validate='many_to_one')
    team_hits = picked.assign(hit=picked.team.eq(picked.winner_team)).groupby('model').hit.sum()
    print(R.groupby('model')[['winner','podium']].agg(['sum','mean']).round(4).to_string(), flush=True)
    print('winning team',team_hits.to_dict(),flush=True)
    print(R.pivot_table(index='season',columns='model',values='winner',aggfunc='sum').to_string(),flush=True)
    for model in ('full_win','full_points'):
        a=R[R.model.eq(model)].set_index(['season','round']).winner
        for ref in ('standings','ens','win_logit'):
            b=R[R.model.eq(ref)].set_index(['season','round']).winner
            print(model,'minus',ref,W.paired_boot(a,b),flush=True)
    probs=joined[joined.model.isin(['full_win','full_null','win_logit','pre_logit'])].copy()
    probs['prob']=-probs.score/(-probs.score).groupby([probs.season,probs['round'],probs.model]).transform('sum')
    loss=probs[probs.order.eq(1)].assign(loss=lambda d:-np.log(d.prob.clip(lower=1e-15))).groupby('model').loss.mean()
    print('winner log-loss',loss.to_dict(),flush=True)
    hits=R.groupby('model').winner.sum()
    gate=hits.full_win>=35 and team_hits.full_win>=team_hits.win_logit and loss.full_win<loss.win_logit and hits.full_win>hits.full_null
    print('INNER GATE',bool(gate),flush=True)
    assert R.season.max()==2022 and R.groupby('model').size().eq(61).all()
    R.to_csv('results/e184_inner_per_race.csv',index=False)
    P.to_csv('results/e184_inner_predictions.csv',index=False)


if __name__=='__main__':
    main()
