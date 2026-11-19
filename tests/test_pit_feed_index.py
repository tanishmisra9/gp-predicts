"""Session/year bounds and metadata-key-only projection."""
from experiments.e229_pit_feed_index import feed_names,index_url


class NoValues(dict):
    def items(self):raise AssertionError('feed values read')
    def values(self):raise AssertionError('feed values read')
    def __getitem__(self,key):raise AssertionError('feed values read')


def test_index():
    payload={'Feeds':NoValues({'TimingData':object(),'PitStopSeries':object(),'TeamRadio':object()}),'Message':'ignored'}
    names,candidates=feed_names(payload)
    assert candidates==['PitStopSeries'] and len(names)==3
    assert feed_names({'Feeds':{'TimingData':{},'TimingAppData':{}}})[1]==[]
    for bad in ({},{'Feeds':[]},{'Feeds':{1:None}}):
        try:feed_names(bad)
        except ValueError:continue
        raise AssertionError('malformed index accepted')
    assert index_url('/static/2019/example_Race/').endswith('/example_Race/Index.json')
    for path in ('/static/2026/example_Race/','/static/2019/example_Qualifying/'):
        try:index_url(path)
        except ValueError:continue
        raise AssertionError('non-pilot source accepted')


if __name__=='__main__':test_index();print('Pit-index metadata-only/year/Race/schema guards pass')
