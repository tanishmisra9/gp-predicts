"""E238 physical-field presence and nonphysical/global/text exclusions."""
import json
from experiments.e238_driver_race_schema import physical_profile,feed_url
from experiments.e230_native_score_schema import project


def body(payloads):
    return ('\n'.join('00:00:00.000'+json.dumps(p) for p in payloads)+'\n').encode()


def test_schema():
    roster={str(i) for i in range(1,11)}
    actual=[{'Lines':{d:{'FuelRemaining':100-k} for d in roster}} for k in range(10)]
    summary,records=physical_profile(body(actual),roster)
    assert summary['physical_qualified'] and summary['physical_leaves']==100 and len(summary['physical_car_keys'])==10
    for payload in [ {'Lines':{d:{'Position':int(d),'LapNumber':5} for d in roster}},
        {'FuelRemaining':10,'Lines':{d:{'Position':int(d)} for d in roster}},
        {'Lines':{d:{'FuelRemaining':True} for d in roster}},
        {'Lines':{d:{'FuelRemaining':'unknown'} for d in roster}},
        {'Lines':{d:{'FuelRemaining':float('inf')} for d in roster}},
        {'Message':{'Lines':{d:{'FuelRemaining':99} for d in roster}}}]:
        assert not physical_profile(body([payload]*20),roster)[0]['physical_qualified']
    assert not physical_profile(b'',roster)[0]['physical_qualified']
    assert not physical_profile(b'garbage\n',roster)[0]['physical_qualified']
    class Poison(dict):
        def __getitem__(self,key):
            if key=='Message': raise AssertionError('text value accessed')
            return super().__getitem__(key)
    assert project(Poison(Message='ignored',FuelRemaining=3),roster)[0][0]['value']==3
    assert feed_url('/static/2019/x_Race/').endswith('/DriverRaceInfo.jsonStream')
    for path in ('/static/2026/x_Race/','/static/2019/x_Qualifying/'):
        try: feed_url(path)
        except ValueError: pass
        else: raise AssertionError('unregistered source accepted')
    print('E238 physical/global/position/boolean/nonfinite/text/empty/source scope checks pass')


if __name__=='__main__':
    test_schema()
