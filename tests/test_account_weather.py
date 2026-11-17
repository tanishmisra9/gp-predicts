"""Account failure cannot trigger acquisition or expose exception contents."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from experiments import e197_account_weather as experiment


def test_account():
    with TemporaryDirectory() as directory:
        root=Path(directory); (root/'results').mkdir()
        with patch.object(experiment,'ROOT',root), patch.dict('os.environ',{},clear=True), patch.object(experiment,'get_token') as login, patch.object(experiment,'weather_source') as acquire:
            experiment.main()
            login.assert_not_called(); acquire.assert_not_called()
            assert json.loads((root/'results/e197_weather_source_summary.json').read_text())['error_type']=='MissingExistingCredentials'
        env={'F1_USERNAME':'test-placeholder','F1_PASSWORD':'test-placeholder'}
        with patch.object(experiment,'ROOT',root), patch.dict('os.environ',env,clear=True), patch.object(experiment,'get_token',side_effect=RuntimeError('private-sentinel')), patch.object(experiment,'weather_source') as acquire:
            experiment.main()
            acquire.assert_not_called()
            text=(root/'results/e197_weather_source_summary.json').read_text()
            assert 'private-sentinel' not in text and json.loads(text)['error_type']=='RuntimeError'
        with patch.dict('os.environ',env,clear=True), patch.object(experiment,'get_token') as login, patch.object(experiment,'weather_source') as acquire:
            experiment.main()
            login.assert_called_once_with(method='password')
            acquire.assert_called_once_with(experiment=197,authenticated=True)


if __name__=='__main__':
    test_account()
    print('missing/failed account stops acquisition; errors sanitized; success dispatches fixed pilot')
