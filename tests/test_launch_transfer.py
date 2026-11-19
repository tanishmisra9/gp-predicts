"""Strict earlier-source history, duplicates/year rejection and bracket limits."""
import numpy as np
import pandas as pd
from experiments.e234_launch_transfer import history,brackets


def test_history():
    source=pd.DataFrame([dict(season=2019,round=r,driver=d,response=v) for r in range(1,8) for d,v in [('A',.1),('B',-.1)]])
    base=history(source)
    altered=source.copy(); altered.loc[altered['round'].ge(5),'response']=50
    again=history(altered)
    np.testing.assert_array_equal(base.loc[base['round'].le(5),'prediction'],again.loc[again['round'].le(5),'prediction'])
    prefix=history(source[source['round'].lt(5)])
    np.testing.assert_array_equal(base.loc[base['round'].lt(5),'prediction'],prefix.prediction)
    earlier=source.copy(); earlier.loc[earlier['round'].eq(1)&earlier.driver.eq('A'),'response']=2
    assert history(earlier).loc[base['round'].eq(5),'prediction'].ne(base.loc[base['round'].eq(5),'prediction']).any()
    for invalid in [pd.concat([source,source.iloc[:1]]),source.assign(season=2020)]:
        try: history(invalid)
        except AssertionError: pass
        else: raise AssertionError('source identity accepted')
    missing=history(source.assign(response=np.nan))
    assert missing.prediction.eq(0).all() and missing.past_count.eq(0).all()
    origin=pd.Timestamp('2019-01-01')
    c=pd.DataFrame(dict(Date=origin+pd.to_timedelta([0,.3,.6],unit='s'),Speed=[0,20,40]))
    assert brackets(c,origin,0)==(.3,.6)
    print('E234 strict-past/current-future/prefix/earlier-response/identity/missing/bracket checks pass')


if __name__=='__main__':
    test_history()
