"""E217 independent cache projection and structured flag identities/count gate."""
import json
from pathlib import Path
from unittest.mock import patch

import fastf1
import pandas as pd

import f1.ff1
from experiments.e190_traffic_feasibility import PILOT,cached_session
from experiments.e217_warning_source import FIELDS


def test_source():
    result=json.loads(Path('results/e217_warning_summary.json').read_text())
    assert [(r['season'],r['round']) for r in result['sessions']]==PILOT
    table=pd.read_csv('results/e217_structured_flags.csv',dtype={'RacingNumber':str})
    table.Time=pd.to_datetime(table.Time,utc=True)
    saved=pd.read_csv('results/e217_warning_rows.csv',dtype={'RacingNumber':str})
    saved.Time=pd.to_datetime(saved.Time,utc=True)
    roster_file=pd.read_csv('results/e217_driver_numbers.csv',dtype={'RacingNumber':str})
    assert list(table)==FIELDS+['season','round'] and list(saved)==FIELDS+['driver','season','round']
    assert 'Message' not in table and 'Message' not in saved
    fastf1.Cache.offline_mode(True); good=0; supported=0; invalid=0
    for report in result['sessions']:
        season=report['season']; round_=report['round']; assert report['available']
        session=cached_session(fastf1,season,round_)
        with patch('requests.adapters.HTTPAdapter.send',side_effect=RuntimeError('warning audit HTTP forbidden')):
            feed=fastf1._api.race_control_messages(session.api_path)
        expected=pd.DataFrame({k:feed[k] for k in FIELDS})
        expected.Time=pd.to_datetime(expected.Time,utc=True)
        actual=table[table['round'].eq(round_)].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual,expected.assign(season=season,round=round_),check_dtype=False)
        raw=pd.DataFrame(session.laps)
        names={}
        for number,driver in raw[['DriverNumber','Driver']].dropna().itertuples(index=False,name=None):
            number=str(number)
            assert number not in names or names[number]==driver
            names[number]=driver
        recorded=roster_file[roster_file['round'].eq(round_)]
        assert dict(zip(recorded.RacingNumber,recorded.driver))==names
        distinct=[]; seen=set()
        for record in expected.to_dict('records'):
            key=tuple(None if pd.isna(record[c]) else record[c] for c in FIELDS)
            if key not in seen:
                distinct.append(record); seen.add(key)
        selected=[]; flag_rows=0; driver_rows=0; warning_rows=0; unassigned=0; counts={}
        for record in distinct:
            category=record['Category']; flag=record['Flag']; scope=record['Scope']
            flag_rows+=int(category=='Flag'); driver_rows+=int(category=='Flag' and scope=='Driver')
            if pd.notna(flag):
                key=(category,flag,scope); counts[key]=counts.get(key,0)+1
            if flag=='BLACK AND WHITE':
                warning_rows+=1
                if category=='Flag' and scope=='Driver' and pd.notna(record['Time']) and record['RacingNumber'] in names:
                    selected.append(dict(**record,driver=names[record['RacingNumber']],season=season,round=round_))
                else:
                    unassigned+=1
        expected_warnings=pd.DataFrame(selected,columns=FIELDS+['driver','season','round'])
        expected_warnings.Time=pd.to_datetime(expected_warnings.Time,utc=True)
        current=saved[saved['round'].eq(round_)].reset_index(drop=True)
        pd.testing.assert_frame_equal(current,expected_warnings,check_dtype=False)
        for key,value in dict(rows=len(expected),distinct_rows=len(distinct),flag_rows=flag_rows,
            driver_flag_rows=driver_rows,warning_rows=warning_rows,assigned_warnings=len(selected),unassigned_warnings=unassigned).items():
            assert report[key]==value
        actual_counts={(r['Category'],r['Flag'],r['Scope']):r['count'] for r in report['flag_schema']}
        assert actual_counts==counts
        for request in report['requests']:
            assert request['url'].endswith(session.api_path+'RaceControlMessages.jsonStream')
        good+=int(len(expected)>=10 and flag_rows>=5 and driver_rows>=1)
        supported+=int(len(selected)>=1); invalid+=unassigned
    assert result['gate']==bool(good==3 and supported>=2 and invalid==0)


if __name__=='__main__':
    test_source()
    print('E217 independent all3cache projection/deduplication/car identities/structured counts and gate PASS')
