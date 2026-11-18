"""One fixed winner model with certified forecasts × strictly earlier driver weather response."""
import json
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd

import f1.walkforward as W
from experiments.e168_winner import win_score
from f1.data import DATA, ROOT, load_results
from f1.features import build_features, pre_weekend_cols

NEW=['forecast_wet_driver_gain','forecast_heat_driver_gain']
KEY=['season','round']


def load_forecasts():
    rows=[]
    for path in sorted((ROOT/'results/e223_receipts').glob('*.json')):
        row=json.loads(path.read_text()); plan=row['plan']
        if plan['season']<2019: continue
        assert 2019<=plan['season']<=2022 and row['complete']
        times=[]
        for obj in row['objects']:
            times.append(parsedate_to_datetime(obj['receipt']['last_modified']))
            times.extend(parsedate_to_datetime(f['last_modified']) for f in obj['fields'].values())
        rows.append(dict(season=plan['season'],round=plan['round'],issue=plan['initialization'],
                         available=max(times).isoformat(),forecast_temp=row['forecast_temp_c'],
                         forecast_rain=row['forecast_precip_mm']))
    out=pd.DataFrame(rows)
    assert len(out)==82 and not out.duplicated(KEY).any()
    return out


def add_weather(X,actual,forecasts,calendar):
    assert not X.duplicated(KEY+['code']).any() and X.season.between(2018,2022).all()
    assert X.index.equals(pd.RangeIndex(len(X)))
    assert not actual.duplicated(KEY).any() and not calendar.duplicated(KEY).any()
    events=calendar[KEY+['date']].copy()
    dates=pd.to_datetime(events.date,utc=True)
    assert dates.dt.dayofweek.eq(6).all()
    events['cutoff']=dates-pd.Timedelta(days=3)
    legal=forecasts.merge(events[KEY+['cutoff']],on=KEY,validate='many_to_one')
    assert not legal.duplicated(KEY+['issue','available']).any()
    legal['issue']=pd.to_datetime(legal.issue,utc=True); legal['available']=pd.to_datetime(legal.available,utc=True)
    legal=legal[legal.issue.lt(legal.cutoff)&legal.available.le(legal.cutoff)&legal.issue.le(legal.available)]
    assert np.isfinite(legal[['forecast_temp','forecast_rain']].to_numpy()).all()
    legal=legal.sort_values(['issue','available']).drop_duplicates(KEY,keep='last')
    current=X[KEY].merge(legal[KEY+['forecast_temp','forecast_rain']],on=KEY,how='left',validate='many_to_one')
    assert current.loc[X.season.ge(2019).to_numpy(),['forecast_temp','forecast_rain']].notna().all().all()
    history=X[KEY+['k','code','order','d10_order']].merge(actual[KEY+['rain_act','temp_act']],on=KEY,how='left',validate='many_to_one')
    history['gain']=history.d10_order.fillna(11)-history.order
    history['wet']=(history.rain_act>1).astype(float).where(history.rain_act.notna())
    history['heat']=((history.temp_act-20)/10).clip(-2,2)
    out=X.copy()
    for feature in NEW: out[feature]=0.
    for k,group in X.groupby('k',sort=True):
        past=history[history.k.lt(k)].copy()
        past['weight']=.5**((k-past.k)/20)
        mu=(past.weight*past.gain).groupby(past.code).sum()/(past.weight.groupby(past.code).sum()+5)
        centered=past.gain-past.code.map(mu)
        wet_weights=past.weight*past.wet
        wet=(wet_weights*centered).groupby(past.code).sum()/(wet_weights.groupby(past.code).sum()+5)
        heat=(past.weight*past.heat*centered).groupby(past.code).sum()/((past.weight*past.heat**2).groupby(past.code).sum()+5)
        rain=current.loc[group.index,'forecast_rain'].clip(0,1).fillna(0)
        temp=((current.loc[group.index,'forecast_temp']-20)/10).clip(-2,2).fillna(0)
        out.loc[group.index,NEW[0]]=group.code.map(wet).fillna(0).to_numpy()*rain.to_numpy()
        out.loc[group.index,NEW[1]]=group.code.map(heat).fillna(0).to_numpy()*temp.to_numpy()
    return out


def inputs():
    df=load_results().loc[lambda d:d.season.le(2022)]
    pace=pd.read_csv(DATA/'racepace_ff1.csv').loc[lambda d:d.season.le(2022)]
    X=build_features(df,pace)
    actual=pd.read_csv(DATA/'weather.csv').loc[lambda d:d.season.le(2022)]
    calendar=df[KEY+['date']].drop_duplicates()
    return X,actual,load_forecasts(),calendar


def main():
    original,actual,forecasts,calendar=inputs(); base=pre_weekend_cols(original)
    X=add_weather(original,actual,forecasts,calendar); F=base+NEW; W.assert_pre_weekend(F)
    W.MODELS.update(
        standings=lambda tr,te,F:(-te.d_season_pts.fillna(0)*100+te.d10_order.fillna(11)*1e-3).to_numpy(),
        pre_logit=lambda tr,te,F:W.m_logit_pod(tr,te,base,C=.03,hl=100),
        win_logit=lambda tr,te,F:win_score(tr,te,base),
        weather_win=lambda tr,te,F:win_score(tr,te,F),
        weather_null=lambda tr,te,F:win_score(tr,te,F,shuffled=True,seed=224))
    R,P=W.walk_forward(X,[2020,2021,2022],['standings','pre_logit','win_logit','weather_win','weather_null'],
                       F=F,min_train_season=2019,ens={'ens':['standings','pre_logit']},tie=None)
    assert R.groupby('model').size().eq(61).all() and P.groupby('model').size().eq(1220).all()
    old=pd.read_csv(ROOT/'results/e214_inner_predictions.csv',float_precision='round_trip')
    references=['standings','pre_logit','ens','win_logit']
    check=P[P.model.isin(references)].merge(old[old.model.isin(references)],on=KEY+['driver','model'],validate='one_to_one',suffixes=('_new','_old'))
    assert len(check)==4880
    np.testing.assert_allclose(check.score_new,check.score_old,atol=1e-14,rtol=0)
    R.to_csv(ROOT/'results/e224_inner_per_race.csv',index=False); P.to_csv(ROOT/'results/e224_inner_predictions.csv',index=False)
    X[KEY+['code','driver','k']+NEW].to_csv(ROOT/'results/e224_features.csv',index=False)
    scored=P.merge(X[KEY+['driver','win']],on=KEY+['driver'],validate='many_to_one')
    probability=[]
    for (s,r,name),g in scored[scored.model.isin(['weather_win','weather_null','win_logit'])].groupby(KEY+['model']):
        p=(-g.score).clip(lower=1e-12); p=p/p.sum()
        probability.append(dict(season=int(s),round=int(r),model=name,logloss=float(-np.log(p[g.win.eq(1)]).sum()),brier=float(((p-g.win)**2).sum())))
    probability=pd.DataFrame(probability); probability.to_csv(ROOT/'results/e224_inner_probabilities.csv',index=False)
    comparison={}; candidate=R[R.model.eq('weather_win')].set_index(KEY).sort_index().winner
    for name in ('standings','ens','win_logit'):
        ref=R[R.model.eq(name)].set_index(KEY).sort_index().winner
        delta,ci,p=W.paired_boot(candidate,ref)
        comparison[name]=dict(difference=float(delta),ci=ci.tolist(),p_nonpositive=float(p))
    hits=R.groupby('model').winner.sum().astype(int).to_dict()
    result=dict(hits=hits,winner_rate=R.groupby('model').winner.mean().to_dict(),strict_overlap=R.groupby('model').podium.mean().to_dict(),
                by_year=R.pivot_table(index='season',columns='model',values='winner',aggfunc='sum').astype(int).to_dict(),
                probabilities=probability.groupby('model')[['logloss','brier']].mean().to_dict('index'),paired=comparison,
                gate=bool(hits['weather_win']>=37 and all(v['ci'][0]>0 for v in comparison.values()) and hits['weather_win']>hits['weather_null']))
    (ROOT/'results/e224_inner_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
