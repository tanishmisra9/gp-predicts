"""E218: cached position Z spatial observability; no race forecast."""
import json
from unittest.mock import patch

import fastf1
import numpy as np
import pandas as pd

from f1.data import ROOT

PILOT=[(2022,6),(2022,13),(2022,16)]
COLS=['driver','cellx','celly','subset','height','n']


def profiles(cars):
    parts=[]; raw_rows=0; ontrack=0; observed=0; padded=0
    for driver,raw in sorted(cars.items()):
        raw_rows+=len(raw); padded+=int(raw.Time.isna().sum())
        selected=raw[raw.Status.eq('OnTrack')&raw.Time.notna()&raw.Date.notna()].copy()
        ontrack+=len(selected)
        keep=np.isfinite(selected[['X','Y','Z']].to_numpy(float)).all(axis=1)
        selected=selected[keep].drop_duplicates('Date',keep='last').sort_values('Date')
        observed+=len(selected)
        parts.append(selected.assign(driver=str(driver)))
    data=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    if data.empty:
        return pd.DataFrame(columns=COLS),dict(raw_rows=raw_rows,ontrack=ontrack,observed=observed,padded_rows=padded,drivers_with_1000=0,finite_fraction=0.)
    midpoint=data.Date.min()+(data.Date.max()-data.Date.min())/2
    data['half']=data.Date.ge(midpoint).astype(int)
    data['cellx']=np.floor(data.X/500).astype(int); data['celly']=np.floor(data.Y/500).astype(int)
    data['height']=data.Z/10  # documented decimetres for 2020+ sources
    rows=[]
    for subset,minimum in [('full',10),('half0',4),('half1',4)]:
        current=data if subset=='full' else data[data.half.eq(int(subset[-1]))]
        g=current.groupby(['driver','cellx','celly']).height.agg(['median','size']).reset_index()
        g=g[g['size'].ge(minimum)].rename(columns={'median':'height','size':'n'})
        rows.append(g.assign(subset=subset)[COLS])
    table=pd.concat(rows,ignore_index=True)
    counts=data.groupby('driver').size()
    return table,dict(raw_rows=raw_rows,ontrack=ontrack,observed=observed,padded_rows=padded,
        drivers_with_1000=int(counts.ge(1000).sum()),finite_fraction=float(observed/ontrack) if ontrack else 0.)


def evaluate(table):
    keys=['driver','cellx','celly','subset']
    assert not table.duplicated(keys).any() and np.isfinite(table.height.to_numpy(float)).all()
    full=table[table.subset.eq('full')]
    field=full.groupby(['cellx','celly']).height.agg(['median','size'])
    field=field[field['size'].ge(8)]
    span=float(np.quantile(field['median'],.95)-np.quantile(field['median'],.05)) if len(field) else 0.
    common=full.merge(field[['median']],on=['cellx','celly'],how='inner',validate='many_to_one')
    agrees=0
    for _,g in common.groupby('driver'):
        error=g.height-g['median']; centered=error-error.median()
        agrees+=int(len(g)>=40 and centered.abs().median()<=2)
    paired=table[table.subset.eq('half0')].merge(table[table.subset.eq('half1')],on=['driver','cellx','celly'],suffixes=('_a','_b'),validate='one_to_one')
    half=paired.groupby(['cellx','celly']).agg(a=('height_a','median'),b=('height_b','median'),drivers=('driver','size'))
    half=half[half.drivers.ge(8)]
    correlation=None; error=None
    if len(half)>=40:
        delta=half.a-half.b
        error=float((delta-delta.median()).abs().median())
        if half.a.std()>0 and half.b.std()>0:
            correlation=float(half.a.corr(half.b))
    gate=bool(len(field)>=40 and span>=5 and agrees>=10 and len(half)>=40 and
        correlation is not None and correlation>=.9 and error<=1)
    return dict(full_cells=len(field),height_span_m=span,agreeing_drivers=agrees,paired_cells=len(half),
        half_correlation=correlation,half_centered_mae_m=error,shape_gate=gate)


def main():
    import f1.ff1
    fastf1.Cache.offline_mode(True); parts=[]; reports=[]
    for season,round_ in PILOT:
        report=dict(season=season,round=round_,available=False)
        try:
            with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('elevation HTTP forbidden')):
                session=fastf1.get_session(season,round_,'Q')
                assert session.api_path.startswith('/static/2022/') and session.api_path.endswith('_Qualifying/')
                cars=fastf1._api.position_data(session.api_path)
        except Exception as exc:
            report['error_type']=type(exc).__name__
        else:
            table,coverage=profiles(cars); shape=evaluate(table)
            parts.append(table.assign(season=season,round=round_))
            report.update(available=True,**coverage,**shape)
            report['gate']=bool(coverage['drivers_with_1000']>=10 and coverage['finite_fraction']>=.95 and shape['shape_gate'])
        reports.append(report); print('SOURCE',json.dumps(report),flush=True)
    summary=dict(sessions=reports,gate=bool(sum(r.get('gate',False) for r in reports)>=2))
    pd.concat(parts,ignore_index=True).to_csv(ROOT/'results/e218_elevation_cells.csv',index=False) if parts else None
    (ROOT/'results/e218_elevation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('RESULT',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
