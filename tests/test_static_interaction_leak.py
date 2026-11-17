"""Exercise the actual legacy interaction loop without running all-season models."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd

FORM=['t10_order','d10_order','t10_pace_gap']


def interactions(frame):
    module=ast.parse(Path('experiments/u_items.py').read_text())
    loops=[node for node in module.body if isinstance(node,ast.For) and isinstance(node.iter,ast.Tuple)
           and [item.value for item in node.iter.elts if isinstance(item,ast.Constant)]==FORM]
    assert len(loops)==1
    program=compile(ast.Module(body=loops,type_ignores=[]),'<legacy form interactions>','exec')
    scope={'X':frame.copy(),'np':np}
    exec(program,scope)
    return scope['X']


def test_future_imputation():
    x=pd.DataFrame(dict(alt_hi=[1.,0.],temp_hist=[32.,np.nan],
        t10_order=[np.nan,2.],d10_order=[np.nan,3.],t10_pace_gap=[np.nan,.5]))
    before=interactions(x)
    future=pd.DataFrame(dict(alt_hi=[1.],temp_hist=[40.],t10_order=[1000.],d10_order=[2000.],t10_pace_gap=[9000.]))
    after=interactions(pd.concat([x,future],ignore_index=True))
    pd.testing.assert_frame_equal(before,after.iloc[:len(x)],check_exact=True)
    for c in FORM:
        assert pd.isna(before.loc[0,'alt_x_'+c]) and pd.isna(before.loc[0,'temp_x_'+c])
        assert before.loc[1,'alt_x_'+c]==0 and before.loc[1,'temp_x_'+c]==0
    # Reproduce the withdrawn full-table median approach independently.
    old=x.alt_hi*x.t10_order.fillna(x.t10_order.median())
    augmented=pd.concat([x,future],ignore_index=True)
    old_future=augmented.alt_hi*augmented.t10_order.fillna(augmented.t10_order.median())
    assert not old.equals(old_future.iloc[:len(x)])
    pd.testing.assert_frame_equal(before[x.columns],x)


if __name__=='__main__':
    test_future_imputation()
    print('legacy static interactions retain missingness and ignore future value distributions')
