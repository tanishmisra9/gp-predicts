"""Calendar projection and fixed local-window support, no HTTP or decoder."""
from experiments.e223_gefs_history import race_plan, plans
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
from experiments.e223_gefs_history import acquire


def test_plan():
    all_plans=plans()
    assert len(all_plans)==103 and sum(p['season']>=2020 for p in all_plans)==61
    assert all(sum(i['overlap_hours'] for i in p['intervals'])==6 for p in all_plans)
    assert all(p['initialization']<p['cutoff']<p['window_start'] for p in all_plans)
    coords={'catalunya':dict(lat='41.57',lon='2.26111'),'bahrain':dict(lat='26.0325',lon='50.5106')}
    r=dict(season='2022',round='6',date='2022-05-22',circuit='catalunya')
    a=race_plan(r,coords)
    assert a['window_start']=='2022-05-22T10:00:00+00:00'
    assert [(i['end'],i['overlap_hours']) for i in a['intervals']]==[(96,2.),(102,4.)]
    assert race_plan(r | dict(order='20',grid='20',rain_act='999'),coords)==a
    b=race_plan(dict(season='2022',round='1',date='2022-03-20',circuit='bahrain'),coords)
    assert b['window_start']=='2022-03-20T15:00:00+00:00'
    assert [(i['end'],i['overlap_hours']) for i in b['intervals']]==[(102,3.),(108,3.)]
    for bad in (r | {'season':'2023'},r | {'date':'2022-05-21'},r | {'date':'2019-05-12'}):
        try:race_plan(bad,coords)
        except ValueError:continue
        raise AssertionError('source/calendar boundary accepted')


def test_future_receipt_never_reads_values():
    plan=plans()[0]
    response=MagicMock(); response.__enter__.return_value=response
    response.status_code=200
    response.headers={'Content-Length':'100000','ETag':'test', 'Last-Modified':'Wed, 18 Nov 2026 00:00:00 GMT'}
    decoder=SimpleNamespace(CodesInternalError=type('DecoderError',(Exception,),{}))
    with patch.dict('sys.modules',{'eccodes':decoder}), patch('requests.head',return_value=response), \
         patch('requests.get',side_effect=AssertionError('future source read')) as get:
        row=acquire(plan)
    assert not row['complete'] and not get.called
    assert 'uncertified source receipt' in row['error']


if __name__=='__main__':
    test_plan()
    test_future_receipt_never_reads_values()
    print('GEFS local-window/calendar boundary tests ok')
