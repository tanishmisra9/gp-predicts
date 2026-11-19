"""Numeric projection, poisoned text, stream format and source bounds."""
import json
from experiments.e230_native_score_schema import project,stream_profile,feed_url


class BlockText(dict):
    def __getitem__(self,key):
        if key in ('Message','Status','Path'):raise AssertionError('text/path accessed')
        return super().__getitem__(key)


def test_projection():
    roster={str(n) for n in range(1,21)}
    payload=BlockText({'Message':object(),'Status':object(),'Path':object(),
                       'Scores':{str(n):{'Score':str(n/2),'Position':n} for n in range(1,21)}})
    values,types,cars=project(payload,roster)
    assert len(values)==40 and types['skipped_text_or_path']==3 and len(cars)==20
    clean={'Scores':payload['Scores'],'Message':'must not export'}
    raw=('00:01:00.000'+json.dumps(clean)+'\r\n').encode()
    summary,records=stream_profile(raw,roster)
    assert summary['qualified'] and summary['records']==1 and summary['numeric_leaves']==40
    assert any(p['candidate_measurement'] and p['path'][-1]=='Score' for p in summary['profiles'])
    assert not any(p['candidate_measurement'] for p in summary['profiles'] if p['path'][-1]=='Position')
    assert 'must not export' not in json.dumps(records)
    assert not stream_profile(b'',roster)[0]['qualified']
    assert not stream_profile(b'bad record',roster)[0]['qualified']
    assert not stream_profile(('00:99:00.000'+json.dumps(clean)).encode(),roster)[0]['qualified']
    constant={'Scores':{str(n):{'Score':0} for n in range(1,21)}}
    assert not stream_profile(('00:00:00:000'+json.dumps(constant)).encode(),roster)[0]['qualified']
    assert project({'Scores':{'1':{'Score':float('nan'),'Flag':True,'Label':'unknown'}}},roster)[0]==[]
    assert feed_url('/static/2019/example_Race/','DriverScore').endswith('/DriverScore.jsonStream')
    for path,feed in [('/static/2026/example_Race/','DriverScore'),('/static/2019/example_Qualifying/','SPFeed'),('/static/2019/example_Race/','TeamRadio')]:
        try:feed_url(path,feed)
        except ValueError:continue
        raise AssertionError('unregistered endpoint accepted')


if __name__=='__main__':test_projection();print('Native numeric projection/poisoned text/stream/source-boundary tests pass')
