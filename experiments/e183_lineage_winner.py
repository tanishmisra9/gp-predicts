"""E183: fixed historical constructor lineage, inner-only winner test."""
import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.b6_adaptation_gate import LINEAGE
from experiments.e168_winner import win_score
from f1.data import DATA, load_results
from f1.features import build_features, pre_weekend_cols


def lineage_features(df, pace):
    assert df.season.le(2022).all(), 'E183 is inner-only'
    keys = ['season', 'round', 'driver']
    assert not df.duplicated(keys).any()
    raw = build_features(df, pace)
    base = pre_weekend_cols(raw)
    team = [c for c in base if c.startswith(('t3_', 't10_', 't_season_', 't_circ_'))]
    mapped = build_features(df.assign(team=df.team.replace(LINEAGE)), pace)
    added = mapped[keys + team].rename(columns={c: 'lineage_' + c for c in team})
    out = raw.merge(added, on=keys, validate='one_to_one')
    candidate = ['lineage_' + c if c in team else c for c in base]
    W.assert_pre_weekend(base + candidate)
    return out, base, candidate


def main():
    df = load_results().loc[lambda d: d.season.le(2022)]
    pace = pd.read_csv(DATA / 'racepace_ff1.csv').loc[lambda d: d.season.le(2022)]
    X, base, candidate = lineage_features(df, pace)
    W.MODELS.update(
        standings=lambda tr, te, F: (-te.d_season_pts.fillna(0)*100 + te.d10_order.fillna(11)*1e-3).to_numpy(),
        pre_logit=lambda tr, te, F: W.m_logit_pod(tr, te, base, C=.03, hl=100),
        win_logit=lambda tr, te, F: win_score(tr, te, base),
        lineage_win=lambda tr, te, F: win_score(tr, te, candidate),
        lineage_null=lambda tr, te, F: win_score(tr, te, candidate, shuffled=True),
    )
    R, P = W.walk_forward(X, [2020, 2021, 2022], ['standings', 'pre_logit', 'win_logit', 'lineage_win', 'lineage_null'],
                          F=base + candidate, ens={'ens': ['standings', 'pre_logit']}, tie=None)
    joined = P.merge(df[['season','round','driver','team','order']], on=['season','round','driver'], validate='many_to_one')
    truth = df[df.order.eq(1)][['season','round','team']].rename(columns={'team': 'winner_team'})
    picked = joined.sort_values(['season','round','model','score','driver']).drop_duplicates(['season','round','model'])
    picked = picked.merge(truth, on=['season','round'], validate='many_to_one')
    team_hits = picked.assign(hit=picked.team.eq(picked.winner_team)).groupby('model').hit.sum()
    print(R.groupby('model')[['winner','podium']].agg(['sum','mean']).round(4).to_string(), flush=True)
    print('winning team', team_hits.to_dict(), flush=True)
    print(R.pivot_table(index='season', columns='model', values='winner', aggfunc='sum').to_string(), flush=True)
    a = R[R.model.eq('lineage_win')].set_index(['season','round']).winner
    for ref in ('standings','ens','win_logit'):
        b = R[R.model.eq(ref)].set_index(['season','round']).winner
        print('lineage_win minus', ref, W.paired_boot(a, b), flush=True)
    probs = joined[joined.model.isin(['lineage_win','lineage_null','win_logit','pre_logit'])].copy()
    probs['prob'] = -probs.score / (-probs.score).groupby([probs.season, probs['round'], probs.model]).transform('sum')
    loss = probs[probs.order.eq(1)].assign(loss=lambda d: -np.log(d.prob.clip(lower=1e-15))).groupby('model').loss.mean()
    print('winner log-loss', loss.to_dict(), flush=True)
    hits = R.groupby('model').winner.sum()
    gate = hits.lineage_win >= 35 and team_hits.lineage_win >= team_hits.win_logit and loss.lineage_win < loss.win_logit and hits.lineage_win > hits.lineage_null
    print('INNER GATE', bool(gate), flush=True)
    assert R.season.max() == 2022 and R.groupby('model').size().eq(61).all()
    R.to_csv('results/e183_inner_per_race.csv', index=False)
    P.to_csv('results/e183_inner_predictions.csv', index=False)


if __name__ == '__main__':
    main()
