"""Safe subset of E69 context; no final-season calendar metadata."""
import numpy as np


CONTEXT_FEATURES = ['gap_leader','tgap_leader','lead_margin']


def add_context(X):
    out=X.copy()
    out['gap_leader']=1-out.d_season_pts
    out['tgap_leader']=1-out.t_season_pts
    second=out.groupby('k').d_season_pts.transform(
        lambda s:s.nlargest(2).iloc[-1] if s.notna().sum()>1 else np.nan)
    out['lead_margin']=1-second
    return out
