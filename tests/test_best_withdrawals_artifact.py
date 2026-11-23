"""E273 independent raw decoding and indexed per-car prefix reconstruction."""
import datetime
import hashlib
import json
from collections import Counter
from unittest.mock import patch

import fastf1
import fastf1._api as api
from fastf1.internals.f1account import LivetimingAuthAdapter

import f1.ff1
from f1.data import ROOT


def integer_time(text, native=False):
    clock, fraction = text.split('.')
    parts = clock.split(':')
    assert len(parts) == (3 if native else 2) and len(fraction) == 3 and fraction.isdigit()
    assert all(p.isdigit() for p in parts)
    if native:
        assert all(len(p) == 2 for p in parts)
        hours, minutes, seconds = map(int, parts)
    else:
        assert 1 <= len(parts[0]) <= 2 and len(parts[1]) == 2
        hours = 0; minutes, seconds = map(int, parts)
    assert 0 <= seconds < 60 and (not native or 0 <= minutes < 60)
    delta = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=int(fraction))
    result = delta.days*86400000+delta.seconds*1000+delta.microseconds//1000
    assert native or result > 0
    return result


def independent_projection(body, roster):
    updates = []; count = 0; last = -1
    for text in body.decode('utf-8-sig').split('\n'):
        text = text.removesuffix('\r')
        if not text:
            continue
        elapsed = integer_time(text[:12], True); assert elapsed >= last; last = elapsed
        data = json.loads(text[12:]); assert isinstance(data, dict)
        for car, row in data.get('Lines', {}).items():
            assert car in roster and isinstance(row, dict)
            a, b = 'BestLapTimes' in row, 'BestLapTime' in row
            if not a and not b:
                continue
            phases = (sorted(row['BestLapTimes']) if isinstance(row.get('BestLapTimes'), dict) else []) if a else None
            if phases is not None:
                assert set(phases) <= {'0', '1', '2'}
            value = row['BestLapTime'].get('Value') if b else None
            value = None if value in [None, ''] else integer_time(value)
            updates.append(dict(record=count, time_ms=elapsed, car=car,
                                phase_present=a, phase_keys=phases, best_present=b, best_ms=value))
        count += 1
    return dict(records=count, updates=updates)


def indexed_analysis(projection, roster):
    events = []; phase_resets = 0; blank_resets = 0; unknown = 0
    updates = Counter({car: 0 for car in roster})
    for car in roster:
        rows = [r for r in projection['updates'] if r['car'] == car]
        phase_entries = [(i, r['phase_keys'][0] if len(r['phase_keys']) == 1 else None)
                         for i, r in enumerate(rows) if r['phase_present']]
        boundaries = [i for j, (i, value) in enumerate(phase_entries)
                      if value is None or value != (phase_entries[j-1][1] if j else None)]
        phase_resets += len(boundaries)
        resets = [i for i, r in enumerate(rows) if r['best_present'] and r['best_ms'] is None]
        blank_resets += len(resets)
        for i, row in enumerate(rows):
            if not row['best_present'] or row['best_ms'] is None:
                continue
            updates[car] += 1
            previous_phases = [(idx, phase) for idx, phase in phase_entries if idx <= i]
            phase = previous_phases[-1][1] if previous_phases else None
            unknown += phase is None
            boundary = max([-1]+[n for n in boundaries+resets if n <= i])
            prior = [r['best_ms'] for r in rows[max(0, boundary):i]
                     if r['best_present'] and r['best_ms'] is not None]
            if not prior or row['best_ms'] <= prior[-1]:
                continue
            old, new = prior[-1], row['best_ms']
            restored = phase is not None and new in prior
            duplicate = restored and any(e['car'] == car and e['phase'] == phase and e['old_ms'] == old
                                         and e['new_ms'] == new and e['restored_seen'] for e in events)
            events.append(dict(record=row['record'], time_ms=row['time_ms'], car=car, code=roster[car], phase=phase,
                               old_ms=old, new_ms=new, restored_seen=restored, duplicate=duplicate))
    events.sort(key=lambda r: (r['record'], list(roster).index(r['car'])))
    unique = [r for r in events if r['restored_seen'] and not r['duplicate']]
    active = sum(n > 0 for n in updates.values())
    return dict(valid_best_updates=sum(updates.values()), unknown_phase_best_updates=unknown,
                phase_resets=phase_resets, blank_best_resets=blank_resets,
                cars_with_best_updates=active, updates_by_car=dict(updates), upward_events=len(events),
                unique_restorations=len(unique), restoration_cars=sorted({r['car'] for r in unique}),
                restoration_codes=sorted({r['code'] for r in unique}), events=events,
                event_gate=bool(sum(updates.values()) >= 100 and active >= 10 and len({r['car'] for r in unique}) >= 2))


def main():
    directory = ROOT / 'results'
    summary = json.loads((directory / 'e273_withdrawal_summary.json').read_text())
    assert summary['source_contract_amended'] and summary['amendment_commit'] == '9066158'
    assert summary['phase_array_policy'] == 'reset_to_unknown'
    saved = json.loads((directory / 'e273_numeric_projection.json').read_text())
    assert hashlib.sha256((directory / 'e273_numeric_projection.json').read_bytes()).hexdigest() == summary['projection_sha256']
    assert hashlib.sha256((directory / 'e270_identity_audit.json').read_bytes()).hexdigest() == summary['identities_sha256']
    roster_map = {(r['season'], r['round']): r['identities'] for r in json.loads((directory / 'e270_identity_audit.json').read_text())}
    assert [(r['season'], r['round']) for r in summary['sources']] == [(2019, 5), (2019, 12), (2019, 14)]
    projections = {(r['season'], r['round']): r for r in saved}
    fastf1.Cache.offline_mode(True); bodies = {}; verified_events = []
    with patch('requests.adapters.HTTPAdapter.send', side_effect=AssertionError('audit HTTP forbidden')) as sends, \
            patch.object(LivetimingAuthAdapter, '_get_token', side_effect=AssertionError('audit token lookup forbidden')) as tokens:
        for receipt in summary['sources']:
            year, rnd = receipt['season'], receipt['round']
            session = fastf1.get_session(year, rnd, 'Q')
            url = api.base_url+session.api_path+'TimingData.jsonStream'
            assert receipt['url'] == url and '/static/2019/' in url and '_Qualifying/' in url
            response = fastf1.Cache.requests_get(url, headers=api.headers)
            assert response.status_code == receipt['cached_status'] and bool(response.from_cache) == receipt['from_cache']
            assert receipt['available'] == bool(response.status_code == 200 and response.from_cache)
            assert receipt['roster'] == roster_map[year, rnd]
            if not receipt['available']:
                assert (year, rnd) not in projections
                continue
            raw = response.content
            assert hashlib.sha256(raw).hexdigest() == receipt['raw_sha256'] and len(raw) == receipt['raw_bytes']
            bodies[url] = receipt['raw_sha256']
            projected = independent_projection(raw, receipt['roster'])
            assert dict(season=year, round=rnd, **projected) == projections[year, rnd]
            assert projected['records'] == receipt['records']
            independently = indexed_analysis(projected, receipt['roster'])
            # Native car iteration order is preserved in the saved projection.
            independently['events'].sort(key=lambda e: (e['record'], next(i for i, row in enumerate(projected['updates'])
                                                                 if row['record'] == e['record'] and row['car'] == e['car'])))
            for field, value in independently.items():
                assert receipt[field] == value, (year, rnd, field)
            verified_events.append(dict(season=year, round=rnd, unique=independently['unique_restorations']))
        assert sends.call_count == tokens.call_count == 0
    reports = summary['sources']
    gate = all(r['available'] for r in reports) and sum(r.get('event_gate', False) for r in reports) >= 2
    gate = gate and sum(r.get('unique_restorations', 0) for r in reports) >= 6
    gate = gate and sum(r.get('unique_restorations', 0) > 0 for r in reports) >= 2
    gate = gate and len({c for r in reports for c in r.get('restoration_codes', [])}) >= 4
    assert gate == summary['source_support_gate']
    assert summary['actual_http_sends'] == summary['token_lookups'] == 0
    assert not any(summary[k] for k in ['deletion_cause_verified', 'completeness_verified',
                                       'historical_publication_time_verified', 'winner_fit', 'outer_evaluated', 'evaluated_2026'])
    (directory / 'e273_independent_audit.json').write_text(json.dumps(dict(events=verified_events,
        source_support_gate=gate, http_sends=0, token_lookups=0, cached_body_hashes_unchanged=bodies), indent=2)+'\n')
    print('E273 independent ALL raw projections/indexed phase-reset-prefix events/counts/gates/hash/scope/zero-HTTP PASS')


if __name__ == '__main__':
    main()
