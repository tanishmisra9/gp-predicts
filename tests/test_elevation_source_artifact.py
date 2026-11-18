"""E218 independent scalar bins/medians and spatial agreement calculations."""
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from unittest.mock import patch

import fastf1
import numpy as np
import pandas as pd

import f1.ff1
from experiments.e218_elevation_source import COLS,PILOT


def quantile(values,p):
    values=sorted(values); index=(len(values)-1)*p; a=math.floor(index); b=math.ceil(index)
    return values[a]+(values[b]-values[a])*(index-a)


def test_source():
    result=json.loads(Path('results/e218_elevation_summary.json').read_text())
    saved=pd.read_csv('results/e218_elevation_cells.csv',dtype={'driver':str},float_precision='round_trip')
    assert [(r['season'],r['round']) for r in result['sessions']]==PILOT
    assert list(saved)==COLS+['season','round']
    fastf1.Cache.offline_mode(True); passes=0
    for report in result['sessions']:
        season=report['season']; round_=report['round']; assert report['available']
        with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('elevation audit HTTP forbidden')):
            session=fastf1.get_session(season,round_,'Q')
            cars=fastf1._api.position_data(session.api_path)
        arrays={}; raw_rows=0; padded=0; ontrack=0; observed=0
        for driver,raw in sorted(cars.items()):
            raw_rows+=len(raw); padded+=int(raw.Time.isna().sum())
            mask=raw.Status.eq('OnTrack')&raw.Time.notna()&raw.Date.notna()
            ontrack+=int(mask.sum())
            subset=raw[mask]
            subset=subset[np.isfinite(subset.X)&np.isfinite(subset.Y)&np.isfinite(subset.Z)]
            subset=subset.drop_duplicates('Date',keep='last').sort_values('Date')
            arrays[str(driver)]=subset; observed+=len(subset)
        first=min(g.Date.min() for g in arrays.values() if len(g))
        last=max(g.Date.max() for g in arrays.values() if len(g)); midpoint=first+(last-first)/2
        groups=defaultdict(list)
        for driver,g in arrays.items():
            for date,x,y,z in g[['Date','X','Y','Z']].itertuples(index=False,name=None):
                cell=(math.floor(x/500),math.floor(y/500)); height=z*.1
                groups[(driver,*cell,'full')].append(height)
                groups[(driver,*cell,'half0' if date<midpoint else 'half1')].append(height)
        values={k:(median(v),len(v)) for k,v in groups.items() if len(v)>=(10 if k[-1]=='full' else 4)}
        expected=pd.DataFrame([dict(zip(['driver','cellx','celly','subset'],k),height=v[0],n=v[1]) for k,v in values.items()])
        actual=saved[saved['round'].eq(round_)]
        keys=['driver','cellx','celly','subset']
        pd.testing.assert_frame_equal(actual[COLS].sort_values(keys).reset_index(drop=True),expected[COLS].sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=1e-12)
        field=defaultdict(list)
        for (driver,x,y,subset),(height,n) in values.items():
            if subset=='full':
                field[(x,y)].append(height)
        field={k:median(v) for k,v in field.items() if len(v)>=8}
        heights=list(field.values()); span=quantile(heights,.95)-quantile(heights,.05) if heights else 0.
        errors=defaultdict(list)
        for (driver,x,y,subset),(height,n) in values.items():
            if subset=='full' and (x,y) in field:
                errors[driver].append(height-field[(x,y)])
        agrees=sum(len(v)>=40 and median(abs(a-median(v)) for a in v)<=2 for v in errors.values())
        half=defaultdict(list)
        for (driver,x,y,subset),(height,n) in values.items():
            other=values.get((driver,x,y,'half1'))
            if subset=='half0' and other is not None:
                half[(x,y)].append((height,other[0]))
        half={k:(median(a for a,b in v),median(b for a,b in v)) for k,v in half.items() if len(v)>=8}
        corr=None; mae=None
        if len(half)>=40:
            pairs=np.array([half[k] for k in sorted(half)])
            a=pairs[:,0]-np.mean(pairs[:,0]); b=pairs[:,1]-np.mean(pairs[:,1])
            if np.sum(a*a)>0 and np.sum(b*b)>0:
                corr=float(np.sum(a*b)/math.sqrt(np.sum(a*a)*np.sum(b*b)))
            delta=pairs[:,0]-pairs[:,1]; centered=delta-np.median(delta)
            mae=float(np.median(abs(centered)))
            energy=np.sort(centered*centered)[::-1]
            print('HALF_DISAGREEMENT',round_,dict(max_abs_m=float(max(abs(centered))),
                top_three_squared_difference_share=float(sum(energy[:3])/sum(energy)) if sum(energy) else 0.),flush=True)
        shape=bool(len(field)>=40 and span>=5 and agrees>=10 and len(half)>=40 and corr is not None and corr>=.9 and mae<=1)
        drivers=sum(len(g)>=1000 for g in arrays.values()); fraction=observed/ontrack if ontrack else 0.
        stats=dict(raw_rows=raw_rows,ontrack=ontrack,observed=observed,padded_rows=padded,drivers_with_1000=drivers,
            finite_fraction=fraction,full_cells=len(field),height_span_m=span,agreeing_drivers=agrees,paired_cells=len(half),
            half_correlation=corr,half_centered_mae_m=mae,shape_gate=shape,gate=bool(drivers>=10 and fraction>=.95 and shape))
        for key,value in stats.items():
            if value is None:
                assert report[key] is None
            elif isinstance(value,float):
                np.testing.assert_allclose(report[key],value,atol=1e-10,rtol=1e-10)
            else:
                assert report[key]==value
        passes+=int(stats['gate'])
    assert result['gate']==bool(passes>=2)


if __name__=='__main__':
    test_source()
    print('E218 independent all3raw profiles/scalar cells/medians/offsets/paired halves and full gate PASS')
