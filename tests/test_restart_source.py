"""Restart pilot handles time boundaries, cohort ranks and pit exclusions."""
import numpy as np
import pandas as pd

from experiments.e206_restart_source import extract_restarts


def fixture():
    laps=[]; samples=[]
    for i in range(8):
        driver=f'd{i}'
        for lap in range(4):
            laps.append(dict(Driver=driver,LapNumber=lap+1,LapStartTime=pd.Timedelta(seconds=60*lap),
                Time=pd.Timedelta(seconds=60*(lap+1)),PitInTime=pd.NaT,PitOutTime=pd.NaT,
                TrackStatus='1' if lap>=2 else '4',IsAccurate=True))
        samples.extend([dict(Driver=driver,Time=pd.Timedelta(seconds=90),Position=i+1),
            dict(Driver=driver,Time=pd.Timedelta(seconds=210),Position=(2-i if i<2 else i+1))])
    status=pd.DataFrame(dict(Time=pd.to_timedelta([0,50,100],unit='s'),Status=['1','4','1']))
    return pd.DataFrame(laps),status,pd.DataFrame(samples)


def test_source():
    laps,status,stream=fixture()
    rows,report=extract_restarts(laps,status,stream)
    assert len(rows)==8 and report[0]['usable'] and rows.gain.sum()==0
    assert rows.set_index('driver').gain.to_dict()==dict(d0=-1,d1=1,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0)
    altered=stream.copy(); altered.Time+=pd.Timedelta(seconds=200)
    pd.testing.assert_frame_equal(extract_restarts(laps,status,pd.concat([stream,altered]))[0],rows)
    pit=laps.copy(); pit['PitInTime']=pd.Series(pd.NaT,index=pit.index,dtype='timedelta64[ns]')
    pit.loc[pit.Driver.eq('d0')&pit.LapNumber.eq(3),'PitInTime']=pd.Timedelta(seconds=175)
    remaining,_=extract_restarts(pit,status,stream)
    assert len(remaining)==7 and remaining.gain.eq(0).all()
    stopped=laps.copy(); stopped=stopped[~(stopped.Driver.eq('d7')&stopped.LapNumber.eq(4))]
    assert len(extract_restarts(stopped,status,stream)[0])==7
    yellow=pd.concat([status,pd.DataFrame(dict(Time=[pd.Timedelta(seconds=200)],Status=['2']))])
    assert extract_restarts(laps,yellow,stream)[0].empty
    assert extract_restarts(laps,status.assign(Status=['1','6','1']),stream)[0].empty
    stale=stream.copy(); stale.loc[stale.Driver.eq('d7')&stale.Time.eq(pd.Timedelta(seconds=90)),'Time']=pd.Timedelta(seconds=64)
    assert len(extract_restarts(laps,status,stale)[0])==7
    exact=stale.copy(); exact.loc[exact.Driver.eq('d7')&exact.Time.eq(pd.Timedelta(seconds=64)),'Time']=pd.Timedelta(seconds=65)
    assert len(extract_restarts(laps,status,exact)[0])==8
    duplicate=stream.copy(); duplicate.loc[duplicate.Driver.eq('d7'),'Position']=7
    assert extract_restarts(laps,status,duplicate)[0].empty
    assert extract_restarts(laps.iloc[:0],status,stream)[0].empty
    try:
        extract_restarts(pd.concat([laps,laps.iloc[:1]]),status,stream)
    except AssertionError:
        pass
    else:
        raise AssertionError('duplicate lap accepted')


if __name__=='__main__':
    test_source()
    print('restart cohort/pit/stoppage/status/age/future-stream and duplicate checks PASS')
