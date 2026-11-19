"""E237 fixed source windows, unchanged tyre/position/status and count gate."""
import numpy as np
import pandas as pd
from experiments.e237_restart_warmup_support import blocks,support


def fixture():
    rows=[]
    for driver in range(12):
        for lap in range(1,7):
            rows.append(dict(Driver=f'D{driver}',LapNumber=lap,LapStartTime=pd.Timedelta(seconds=(lap-1)*90),
                Time=pd.Timedelta(seconds=lap*90),LapTime=pd.Timedelta(seconds=90),Stint=1,Compound='MEDIUM',TyreLife=lap,
                Position=driver+1,PitInTime=pd.NaT,PitOutTime=pd.NaT,TrackStatus='4' if lap==1 else ('41' if lap==2 else '1'),IsAccurate=True))
    status=pd.DataFrame(dict(Time=pd.to_timedelta([0,80,100],unit='s'),Status=['4','4','1']))
    return pd.DataFrame(rows),status


def test_windows():
    laps,status=fixture(); source,reports=blocks(laps,status)
    assert len(source)==12 and source.first_full_lap.eq(3).all() and source.previous_lap.eq(2).all()
    assert len(reports)==1 and reports[0]['sc_seconds']==100
    for field,value in [('Position',99),('Compound','HARD'),('TyreLife',99),('Stint',2),('PitInTime',pd.Timedelta(seconds=300)),('IsAccurate',False),('TrackStatus','2'),('LapTime',pd.Timedelta(seconds=100))]:
        changed=laps.copy(); changed.loc[changed.Driver.eq('D0')&changed.LapNumber.eq(4),field]=value
        assert len(blocks(changed,status)[0])==11,field
    changed=laps.copy(); changed.loc[changed.Driver.eq('D0')&changed.LapNumber.eq(2),'PitOutTime']=pd.Timedelta(seconds=95)
    assert len(blocks(changed,status)[0])==11
    changed=status.copy(); changed.loc[len(changed)]=[pd.Timedelta(seconds=200),'2']
    assert blocks(laps,changed)[0].empty
    assert blocks(laps,status.assign(Status='1'))[0].empty
    assert blocks(laps.iloc[:0],status)[0].empty
    try: blocks(pd.concat([laps,laps.iloc[:1]]),status)
    except AssertionError: pass
    else: raise AssertionError('duplicate lap accepted')
    entries=pd.DataFrame([dict(season=2019,round=r,code=f'D{d}',team=f'T{d//2}') for r in range(1,13) for d in range(12)])
    repeated=pd.concat([source.assign(season=2019,round=r) for r in range(1,13)],ignore_index=True)
    result,pairs=support(repeated,entries)
    assert result['gate'] and result['qualified_races']==12 and result['qualified_team_races']==72 and len(pairs)==72
    result,_=support(repeated[~repeated.driver.isin(['D0','D1'])],entries)
    assert not result['gate'] and result['qualified_races']==0
    result,_=support(repeated.iloc[:0],entries)
    assert not result['gate']
    print('E237 SC episode/stable tyre-position/first raw lap/caution/empty/duplicate/team support checks pass')




def test_missing_cached_fields():
    import json
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import patch
    import experiments.e237_restart_warmup_support as E
    class Broken:
        @property
        def laps(self): raise RuntimeError('cached session has no laps')
    laps,status=fixture()
    entries=pd.DataFrame([dict(season=2018,round=r,code=f'D{d}',team=f'T{d//2}') for r in (1,2) for d in range(12)])
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder)/'results').mkdir()
        with patch.object(E,'ROOT',Path(folder)),patch.object(E,'load_results',return_value=entries),patch.object(E,'cached_session',side_effect=[Broken(),SimpleNamespace(laps=laps,track_status=status)]):
            E.main()
        report=json.loads((Path(folder)/'results/e237_support_summary.json').read_text())
        assert not report['sessions'][0]['available'] and report['sessions'][0]['error_type']=='RuntimeError'
        assert report['sessions'][1]['available']


if __name__=='__main__':
    test_windows()
    test_missing_cached_fields()
