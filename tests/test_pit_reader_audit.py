import hashlib
import json
from experiments.e256_pit_reader_audit import inspect_reader,COMMIT,BLOB
from f1.data import ROOT


def main():
    fixture="""class PitCollection:
 def process_message(self,message):
  if message.topic == 'Lane':
   lane_duration=float(data['Duration'])
   pit=Pit(stop_duration=None)
  elif message.topic == 'Service':
   stop_duration=float(data['PitStopTime'])
   pit=Pit(stop_duration=stop_duration)
"""
    r=inspect_reader(fixture)
    assert r['Lane']==dict(stop_argument='None',lane_input_keys=['Duration'],stop_input_keys=[])
    assert r['Service']['stop_argument']=='stop_duration' and r['Service']['stop_input_keys']==['PitStopTime']
    changed=inspect_reader(fixture.replace('stop_duration=None','stop_duration=unexpected'))
    assert changed['Lane']['stop_argument']!='None'
    report=json.loads((ROOT/'results/e256_pit_reader_audit.json').read_text())
    oldpath=ROOT/'results/e229_pit_feed_index.json'
    assert report['commit']==COMMIT and report['blob_sha1']==BLOB
    assert report['index_sha256']==hashlib.sha256(oldpath.read_bytes()).hexdigest()
    old=json.loads(oldpath.read_text())
    assert [(s['season'],s['round']) for s in report['sources']]==[(2019,5),(2019,12),(2019,14)]
    service={n for n,r in report['reader'].items() if r['stop_argument']!='None'}
    assert sorted(service)==report['service_topics']==['PitStop','PitStopSeries']
    for actual,source in zip(report['sources'],old['sessions']):
        names=set(source['feeds'])
        assert actual['service_topics_present']==sorted(names&service)
        assert actual['lane_present']==('PitLaneTimeCollection' in names)
    assert not any(report[k] for k in ('raw_historical_service_fields_checked','historical_units_verified','winner_fit'))
    print('E256 synthetic branch/mapping falsification and independent saved feed inventory/hash/limited-claim audit PASS')


if __name__=='__main__':main()
