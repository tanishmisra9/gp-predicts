"""E240 matched activity and first-repass duration, fixed next-line labels."""
import pandas as pd
from experiments.e240_native_pass_novelty import compare,summarize


def fixture():
    rows=[]
    for driver,number,position,offset in [('A','1',2,1),('B','2',1,0)]:
        for lap in range(1,5):
            rows.append(dict(Driver=driver,DriverNumber=number,LapNumber=lap,Position=position,
                Time=pd.Timedelta(seconds=lap*100+offset),PitInTime=pd.NaT,PitOutTime=pd.NaT,TrackStatus='1',IsAccurate=True))
    laps=pd.DataFrame(rows)
    status=pd.DataFrame(dict(Time=pd.to_timedelta([0],unit='s'),Status=['1']))
    markers=[dict(echo=True,car='1',counterparts=['2'],seconds=225),dict(echo=True,car='2',counterparts=['1'],seconds=235)]
    return laps,status,markers


def test_compare():
    laps,status,markers=fixture()
    rows=compare(laps,status,markers,(0,500)); matched=[r for r in rows if r['matched']]
    assert len(matched)==1 and matched[0]['old_pass']==0 and matched[0]['native_pass'] and matched[0]['confirmed_transient']
    assert any(r['confirmed_transient'] for r in compare(laps,status,[markers[0],dict(markers[1],seconds=230)],(0,500)))
    assert not any(r['confirmed_transient'] for r in compare(laps,status,[markers[0],dict(markers[1],seconds=229.999)],(0,500)))
    early=[dict(markers[0]),dict(markers[1],seconds=227),dict(markers[1],seconds=240)]
    assert not any(r['confirmed_transient'] for r in compare(laps,status,early,(0,500)))
    assert not any(r['native_pass'] for r in compare(laps,status,[],(0,500)))
    assert not any(r['matched'] for r in compare(laps,status,markers,(210,500)))
    for field,value in [('PitOutTime',pd.Timedelta(seconds=350)),('IsAccurate',False),('TrackStatus','2'),('Time',pd.NaT)]:
        changed=laps.copy();changed.loc[changed.Driver.eq('A')&changed.LapNumber.eq(4),field]=value
        assert not any(r['matched'] for r in compare(changed,status,markers,(0,500))),field
    cautions=pd.DataFrame(dict(Time=pd.to_timedelta([0,250,250],unit='s'),Status=['1','2','1']))
    assert not any(r['matched'] for r in compare(laps,cautions,markers,(0,500)))
    assert not summarize([dict(rows=[])])['gate']
    positive=dict(matched[0],old_pass=1,confirmed_transient=False)
    novel=dict(matched[0])
    sources=[dict(rows=[positive]*20+[novel]*5) for _ in range(2)]+[dict(rows=[positive]*20)]
    assert summarize(sources)['gate']
    assert not summarize([dict(rows=[positive]*20+[novel]*4),dict(rows=[positive]*20+[novel]*5),dict(rows=[positive]*20)])['gate']
    print('E240 activity/pit/caution/source-span/first-repass/empty/gate checks pass')


if __name__=='__main__':
    test_compare()
