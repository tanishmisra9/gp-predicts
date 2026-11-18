"""Frozen unit/time/parameter/nearest-value gates, no decoder or network needed."""
from experiments.e222_gefs_decode import validate


def test_decode_gate():
    m = dict(edition=2, discipline=0, dataDate=20200812, dataTime=1200,
             validityDate=20200816, validityTime=1200, stepUnits=1, endStep=96,
             gridType='regular_ll', parameterCategory=0, parameterNumber=0, units='K',
             typeOfLevel='heightAboveGround', level=2, startStep=96, stepType='instant')
    p = dict(value=300., distance=20.)
    init = '2020-08-12T12:00:00+00:00'
    assert validate(m, p, 'TMP', init)
    for change in ({'dataDate':20200813}, {'validityTime':1800}, {'stepUnits':0}, {'endStep':102},
                   {'units':'C'}, {'level':10}, {'parameterNumber':1}, {'startStep':90}, {'stepType':'accum'}):
        assert not validate(m | change, p, 'TMP', init)
    for change in ({'value':float('nan')}, {'value':9999.}, {'distance':80.01}, {'distance':-1}):
        assert not validate(m, p | change, 'TMP', init)
    rain = m | dict(parameterCategory=1, parameterNumber=8, units='kg m**-2',
                    typeOfLevel='surface', startStep=90, stepType='accum')
    assert validate(rain, p | {'value':0}, 'APCP', init)
    assert not validate(rain | {'startStep':0}, p, 'APCP', init)
    assert not validate(rain, p | {'value':-0.01}, 'APCP', init)


if __name__ == '__main__':
    test_decode_gate()
    print('GEFS decoded-field gate tests ok')
