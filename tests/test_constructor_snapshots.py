"""E242 exact snapshot identity/numerics, ignored text and adjusted points."""
import pandas as pd
from experiments.e242_constructor_snapshots import project,compare


def fixture():
    rows=[dict(position=str(i+1),points=str(100-i),wins='0',Constructor=dict(constructorId=f'team{i}',name='unused',url='unused')) for i in range(10)]
    return dict(MRData=dict(offset='0',total='10',StandingsTable=dict(season='2020',round='17',StandingsLists=[dict(season='2020',round='17',ConstructorStandings=rows)])))


def test_schema():
    payload=fixture();rows=project(payload,2020,17)
    assert len(rows)==10 and rows[0]['constructor']=='team0'
    class Poison(dict):
        def __getitem__(self,key):
            if key in ('name','url'):raise AssertionError('irrelevant field accessed')
            return super().__getitem__(key)
    payload['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings'][0]['Constructor']=Poison(constructorId='team0',name='unused',url='unused')
    assert project(payload,2020,17)==rows
    for kind in ('year','round','offset','duplicate','nan','boolean','rank','wins'):
        bad=fixture(); data=bad['MRData'];table=data['StandingsTable'];r=table['StandingsLists'][0]['ConstructorStandings']
        if kind=='year':table['season']='2026'
        elif kind=='round':table['StandingsLists'][0]['round']='18'
        elif kind=='offset':data['offset']='10'
        elif kind=='duplicate':r[1]['Constructor']['constructorId']='team0'
        elif kind=='nan':r[0]['points']='nan'
        elif kind=='boolean':r[0]['wins']=True
        elif kind=='rank':r[0]['position']='11'
        elif kind=='wins':r[0]['wins']='18'
        try:project(bad,2020,17)
        except ValueError:pass
        else:raise AssertionError(kind)
    history=pd.DataFrame([dict(season=2020,round=17,driver=f'd{i}',team=f'team{i}',points=100-i,s_points=0.) for i in range(10)])
    history.loc[history.team.eq('team1'),'points']=110
    compared,inversions=compare(rows,history)
    assert compared[1]['official_minus_combined']==-11 and ['team0','team1'] in inversions
    history.loc[history.team.eq('team1'),'s_points']=2
    assert compare(rows,history)[0][1]['official_minus_combined']==-13
    print('E242 source identity/pagination/numeric/text-skip/adjusted-point/inversion checks pass')


if __name__=='__main__':
    test_schema()
