"""Real inner source-prefix invariance and independent scalar weather histories."""
import math
import numpy as np
import pandas as pd
from experiments.e224_weather_winner import NEW,KEY,inputs,add_weather


def test_real_features():
    X,actual,forecasts,calendar=inputs()
    out=add_weather(X,actual,forecasts,calendar)
    for season,rnd in ((2020,6),(2022,10)):
        k=int(X.loc[X.season.eq(season)&X['round'].eq(rnd),'k'].iloc[0])
        events=X[KEY+['k']].drop_duplicates()
        a=actual.merge(events,on=KEY,validate='one_to_one')
        a.loc[a.k.ge(k),['rain_act','temp_act']]=[999.,999.]
        altered=X.copy(); altered.loc[altered.k.ge(k),['order','d10_order']]=[999.,777.]
        changed=add_weather(altered,a,forecasts,calendar)
        np.testing.assert_array_equal(out.loc[X.k.le(k),NEW],changed.loc[X.k.le(k),NEW])
        truncated=add_weather(X[X.k.le(k)].copy(),actual,forecasts,calendar)
        np.testing.assert_array_equal(out.loc[X.k.le(k),NEW],truncated[NEW])
        f=forecasts.merge(events,on=KEY,validate='one_to_one')
        f.loc[f.k.gt(k),['forecast_temp','forecast_rain']]=[999.,999.]
        changed=add_weather(X,actual,f,calendar)
        np.testing.assert_array_equal(out.loc[X.k.le(k),NEW],changed.loc[X.k.le(k),NEW])
    late=forecasts.copy(); late['available']='2026-11-18T00:00:00Z'; late[['forecast_temp','forecast_rain']]=999.
    np.testing.assert_array_equal(out[NEW],add_weather(X,actual,pd.concat([forecasts,late],ignore_index=True),calendar)[NEW])
    observed={(int(r.season),int(r.round)):(float(r.rain_act),float(r.temp_act)) for r in actual.itertuples(index=False)}
    predicted={(int(r.season),int(r.round)):(float(r.forecast_rain),float(r.forecast_temp)) for r in forecasts.itertuples(index=False)}
    history={}; expected=[]
    for k in sorted(X.k.unique()):
        group=X[X.k.eq(k)]
        for r in group.itertuples():
            past=history.get(r.code,[])
            w=[math.exp2(-(k-p[0])/20) for p in past]
            mu=sum(a*p[1] for a,p in zip(w,past))/(sum(w)+5)
            wetnum=wetden=heatnum=heatden=0.
            for a,p in zip(w,past):
                if math.isfinite(p[2]):
                    wet=float(p[2]>1); wetnum+=a*wet*(p[1]-mu); wetden+=a*wet
                if math.isfinite(p[3]):
                    heat=max(-2.,min(2.,(p[3]-20)/10))
                    heatnum+=a*heat*(p[1]-mu); heatden+=a*heat*heat
            rain,temp=predicted.get((r.season,r.round),(0.,20.))
            values=[wetnum/(wetden+5)*max(0.,min(1.,rain)),heatnum/(heatden+5)*max(-2.,min(2.,(temp-20)/10))]
            np.testing.assert_allclose(out.loc[r.Index,NEW].to_numpy(float),values,rtol=0,atol=1e-12)
            expected.append(values)
        for r in group.itertuples():
            rain,temp=observed.get((r.season,r.round),(float('nan'),float('nan')))
            form=float(r.d10_order) if math.isfinite(float(r.d10_order)) else 11.
            history.setdefault(r.code,[]).append((k,form-r.order,rain,temp))
    assert len(expected)==2060
    print('Real prefix/future actual+label/late+future forecast invariance and all2060 scalar histories pass')


if __name__=='__main__': test_real_features()
