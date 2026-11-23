"""E273 numeric best-time withdrawal support; cache only, no deletion labels."""
import hashlib
import json
import re
from unittest.mock import patch

from f1.data import ROOT

PILOT = [(2019, 5), (2019, 12), (2019, 14)]


def milliseconds(value, native=False):
    pattern = r'([0-9]{2}):([0-5][0-9]):([0-5][0-9])\.([0-9]{3})' if native else r'([0-9]{1,2}):([0-5][0-9])\.([0-9]{3})'
    assert isinstance(value, str)
    match = re.fullmatch(pattern, value); assert match, 'invalid clock'
    parts = list(map(int, match.groups()))
    if native:
        hour, minute, second, milli = parts
        return ((hour*60+minute)*60+second)*1000+milli
    minute, second, milli = parts
    result = (minute*60+second)*1000+milli
    assert result > 0, 'nonpositive best time'
    return result


def project(body, roster):
    assert len(roster) == len(set(roster.values())), 'ambiguous car identity'
    rows = []; previous = -1; records = 0
    for line in body.decode('utf-8-sig').splitlines():
        if not line:
            continue
        time = milliseconds(line[:12], native=True)
        assert time >= previous, 'decreasing native clock'
        previous = time
        payload = json.loads(line[12:]); assert isinstance(payload, dict)
        lines = payload.get('Lines', {}); assert isinstance(lines, dict)
        for car, data in lines.items():
            assert car in roster and car.isdigit() and isinstance(data, dict), 'unknown car or malformed patch'
            if 'BestLapTimes' not in data and 'BestLapTime' not in data:
                continue
            phase = data.get('BestLapTimes')
            if 'BestLapTimes' in data:
                assert isinstance(phase, dict) and set(phase) <= {'0', '1', '2'}
            best = data.get('BestLapTime')
            if 'BestLapTime' in data:
                assert isinstance(best, dict)
                value = best.get('Value')
                best_ms = None if value is None or value == '' else milliseconds(value)
            else:
                best_ms = None
            rows.append(dict(record=records, time_ms=time, car=car,
                             phase_present='BestLapTimes' in data,
                             phase_keys=sorted(phase) if phase is not None else None,
                             best_present='BestLapTime' in data, best_ms=best_ms))
        records += 1
    return dict(records=records, updates=rows)


def analyse(projection, roster):
    state = {}; events = []; seen_events = set(); updates = {car: 0 for car in roster}
    counts = dict(valid_best_updates=0, unknown_phase_best_updates=0, phase_resets=0, blank_best_resets=0)
    last_key = (-1, -1); keys = set()
    for row in projection['updates']:
        assert row['car'] in roster and (row['record'], row['car']) not in keys
        keys.add((row['record'], row['car']))
        key = row['record'], row['time_ms']
        assert key >= last_key
        last_key = key
        car = row['car']; current = state.setdefault(car, dict(phase=None, best=None, seen=set()))
        if row['phase_present']:
            phase = row['phase_keys'][0] if len(row['phase_keys']) == 1 else None
            if phase is None or phase != current['phase']:
                current.update(phase=phase, best=None, seen=set()); counts['phase_resets'] += 1
        if not row['best_present']:
            continue
        value = row['best_ms']
        if value is None:
            current.update(best=None, seen=set()); counts['blank_best_resets'] += 1
            continue
        assert isinstance(value, int) and not isinstance(value, bool) and value > 0
        updates[car] += 1; counts['valid_best_updates'] += 1
        counts['unknown_phase_best_updates'] += current['phase'] is None
        old = current['best']
        if old is not None and value > old:
            restored = current['phase'] is not None and value in current['seen']
            signature = car, current['phase'], old, value
            duplicate = restored and signature in seen_events
            events.append(dict(record=row['record'], time_ms=row['time_ms'], car=car,
                               code=roster[car], phase=current['phase'], old_ms=old, new_ms=value,
                               restored_seen=restored, duplicate=duplicate))
            if restored:
                seen_events.add(signature)
        current['best'] = value; current['seen'].add(value)
    unique = [r for r in events if r['restored_seen'] and not r['duplicate']]
    return dict(**counts, cars_with_best_updates=sum(v > 0 for v in updates.values()),
                updates_by_car=updates, upward_events=len(events), unique_restorations=len(unique),
                restoration_cars=sorted({r['car'] for r in unique}),
                restoration_codes=sorted({r['code'] for r in unique}), events=events,
                event_gate=bool(counts['valid_best_updates'] >= 100 and sum(v > 0 for v in updates.values()) >= 10
                                and len({r['car'] for r in unique}) >= 2))


def gate(reports):
    assert [(r['season'], r['round']) for r in reports] == PILOT
    return bool(all(r['available'] for r in reports) and sum(r.get('event_gate', False) for r in reports) >= 2
                and sum(r.get('unique_restorations', 0) for r in reports) >= 6
                and sum(r.get('unique_restorations', 0) > 0 for r in reports) >= 2
                and len({c for r in reports for c in r.get('restoration_codes', [])}) >= 4)


def main():
    import fastf1
    import fastf1._api as api
    import f1.ff1
    from fastf1.internals.f1account import LivetimingAuthAdapter
    output = ROOT / 'results/e273_withdrawal_summary.json'
    assert not output.exists(), 'do not overwrite a measured pilot'
    identities_path = ROOT / 'results/e270_identity_audit.json'
    identities_hash = hashlib.sha256(identities_path.read_bytes()).hexdigest()
    identities = {(r['season'], r['round']): r['identities'] for r in json.loads(identities_path.read_text())}
    fastf1.Cache.offline_mode(True); reports = []; projections = []
    with patch('requests.adapters.HTTPAdapter.send', side_effect=RuntimeError('E273 HTTP forbidden')) as sends, \
            patch.object(LivetimingAuthAdapter, '_get_token', side_effect=RuntimeError('E273 token lookup forbidden')) as tokens:
        for year, rnd in PILOT:
            roster = identities[year, rnd]
            assert len(roster) == len(set(roster.values()))
            session = fastf1.get_session(year, rnd, 'Q')
            assert session.api_path.startswith('/static/2019/') and session.api_path.endswith('_Qualifying/')
            url = api.base_url+session.api_path+'TimingData.jsonStream'
            report = dict(season=year, round=rnd, available=False, url=url, roster=roster)
            response = fastf1.Cache.requests_get(url, headers=api.headers)
            report.update(cached_status=response.status_code, from_cache=bool(getattr(response, 'from_cache', False)))
            if response.status_code == 200 and report['from_cache']:
                body = response.content
                projected = project(body, roster)
                report.update(available=True, raw_bytes=len(body), raw_sha256=hashlib.sha256(body).hexdigest(),
                              records=projected['records'], **analyse(projected, roster))
                projections.append(dict(season=year, round=rnd, **projected))
            reports.append(report)
            print('SOURCE', year, rnd, report['available'], report.get('unique_restorations'), flush=True)
        assert sends.call_count == tokens.call_count == 0
    assert hashlib.sha256(identities_path.read_bytes()).hexdigest() == identities_hash
    result = dict(sources=reports, source_support_gate=gate(reports), actual_http_sends=0, token_lookups=0,
                  identities_sha256=identities_hash, deletion_cause_verified=False, completeness_verified=False,
                  historical_publication_time_verified=False, winner_fit=False, outer_evaluated=False, evaluated_2026=False)
    projection_path = ROOT / 'results/e273_numeric_projection.json'
    with projection_path.open('x') as file:
        json.dump(projections, file, indent=2); file.write('\n')
    result['projection_sha256'] = hashlib.sha256(projection_path.read_bytes()).hexdigest()
    with output.open('x') as file:
        json.dump(result, file, indent=2); file.write('\n')
    print('SOURCE_GATE', result['source_support_gate'], flush=True)


if __name__ == '__main__':
    main()
