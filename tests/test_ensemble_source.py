"""Common-member identities, period/unit bounds, missing-member fallback and cutoff rejection."""
from types import SimpleNamespace
import json
from unittest.mock import MagicMock,patch
from experiments.e226_gefs_ensemble import ROOT,member_url,field_gate,probability,acquire


def test_source():
    sources=[json.loads(p.read_text()) for p in (ROOT/'results/e223_receipts').glob('*.json')]
    sources=[r for r in sources if r['plan']['season']>=2019]
    assert len(sources)==82 and sum(len(r['plan']['intervals']) for r in sources)==159
    assert member_url('https://example/gec00.t12z.f096',1)=='https://example/gep01.t12z.f096'
    for n in (0,21):
        try:member_url('gec00',n)
        except ValueError:continue
        raise AssertionError('unregistered member accepted')
    assert probability(0.,[])==(0.,False)
    rows=[dict(member=m,complete=True,forecast_rain=2. if m<=10 else 0.) for m in range(1,21)]
    assert probability(2.,rows)==(11/21,True)
    rows[0]['complete']=False; assert probability(2.,rows)==(1.,False)
    plan=dict(initialization='2020-08-12T12:00:00+00:00',cutoff='2020-08-13T00:00:00+00:00',
              intervals=[dict(url='https://example/gec00',end=96,overlap_hours=6)])
    m=dict(edition=2,discipline=0,parameterCategory=1,parameterNumber=8,units='kg m**-2',typeOfLevel='surface',
        dataDate=20200812,dataTime=1200,validityDate=20200816,validityTime=1200,stepUnits=1,startStep=90,endStep=96,
        stepType='accum',gridType='regular_ll',perturbationNumber=1)
    p=dict(value=0.,distance=20.)
    assert field_gate(m,p,plan,96,1)
    for change in ({'perturbationNumber':2},{'startStep':0},{'units':'m'},{'dataDate':20200813},{'stepUnits':0}):
        assert not field_gate(m|change,p,plan,96,1)
    for change in ({'value':-1.},{'value':float('nan')},{'distance':81.}):
        assert not field_gate(m,p|change,plan,96,1)
    response=MagicMock();response.__enter__.return_value=response
    response.status_code=200;response.headers={'Content-Length':'123','ETag':'x','Last-Modified':'Wed, 18 Nov 2026 00:00:00 GMT'}
    session=MagicMock();session.__enter__.return_value=session;session.head.return_value=response
    session.get.side_effect=AssertionError('future receipt must not fetch data')
    decoder=SimpleNamespace(CodesInternalError=type('DecoderError',(Exception,),{}))
    with patch.dict('sys.modules',{'eccodes':decoder}),patch('requests.Session',return_value=session): row=acquire(plan,1)
    assert not row['complete'] and not session.get.called
    response.headers['Last-Modified']='Wed, 12 Aug 2020 17:00:00 GMT'
    index=MagicMock();index.__enter__.return_value=index;index.status_code=200
    index.raw.read.return_value=b'1:0:d=2020081212:APCP:surface:90-96 hour acc fcst:\n'
    ignored=MagicMock();ignored.__enter__.return_value=ignored;ignored.status_code=200;ignored.headers={}
    ignored.raw.read.side_effect=AssertionError('ignored range body read')
    session.get.side_effect=[index,ignored]
    with patch.dict('sys.modules',{'eccodes':decoder}),patch('requests.Session',return_value=session): row=acquire(plan,1)
    assert not row['complete'] and not ignored.raw.read.called


if __name__=='__main__':test_source();print('Ensemble source/member/fallback/cutoff tests ok')
