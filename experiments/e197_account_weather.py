"""E197: use the project's existing account helper for a fixed historical pilot."""
import json
import os

import fastf1
from fastf1.internals.f1account import get_token

from experiments.e196_weather_source import main as weather_source
from f1.data import ROOT


def main():
    import f1.ff1  # project cache setup; helper already loaded only the two F1 keys
    error=None
    if not all(os.environ.get(key) for key in ('F1_USERNAME','F1_PASSWORD')):
        error='MissingExistingCredentials'
    else:
        fastf1.Cache.offline_mode(False)
        try:
            get_token(method='password')
        except Exception as exc:
            error=type(exc).__name__
    if error is not None:
        result=dict(sessions=[],gate=False,authenticated=False,error_type=error)
        (ROOT/'results/e197_weather_source_summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        return
    weather_source(experiment=197,authenticated=True)


if __name__=='__main__':
    main()
