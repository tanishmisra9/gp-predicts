"""E217 uses structured flag identity only, never free-text or inferred car IDs."""
import pandas as pd

from experiments.e217_warning_source import FIELDS,allowed_urls,observe,structured


def test_flags():
    class NoText(dict):
        def __getitem__(self,key):
            assert key!='Message','free text accessed'
            return super().__getitem__(key)
    feed=NoText(Time=['2019-09-08T13:20:00Z']*3,Category=['Flag','Flag','Other'],
        Flag=['BLACK AND WHITE','BLUE',None],Scope=['Driver','Driver',None],RacingNumber=['16','44',None],Lap=[20,20,None])
    frame=structured(feed); assert list(frame)==FIELDS
    roster=pd.DataFrame(dict(RacingNumber=['16','44'],driver=['LEC','HAM']))
    warnings,report=observe(frame,roster)
    assert len(warnings)==report['assigned_warnings']==1 and warnings.driver.iloc[0]=='LEC'
    assert report['unassigned_warnings']==0 and report['driver_flag_rows']==2
    assert observe(pd.concat([frame,frame.iloc[:1]]),roster)[1]['assigned_warnings']==1
    bad=frame.copy(); bad.loc[0,'RacingNumber']='99'
    assert observe(bad,roster)[1]['unassigned_warnings']==1
    bad=frame.copy(); bad.loc[0,'Time']=pd.NaT
    assert observe(bad,roster)[1]['unassigned_warnings']==1
    bad=frame.copy(); bad.loc[0,'Scope']='Track'
    assert observe(bad,roster)[1]['unassigned_warnings']==1
    assert observe(frame.iloc[:0],roster)[1]['assigned_warnings']==0
    path='/static/2019/2019-09-08_Italian_Grand_Prix/2019-09-08_Race/'
    assert len(allowed_urls(path))==2
    for path in [path.replace('2019','2026'),path.replace('_Race/','_Practice_1/')]:
        try:
            allowed_urls(path)
        except AssertionError:
            pass
        else:
            raise AssertionError('nonregistered source endpoint accepted')


if __name__=='__main__':
    test_flags()
    print('warning structured-only projection, identities, deduplication, missingness and endpoint guards PASS')
