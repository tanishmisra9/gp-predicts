"""Post-inspection technical source audit; no event records, messages or model."""
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

from f1.data import ROOT


def facts(source):
    session = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Session')
    method = next(n for n in session.body if isinstance(n, ast.FunctionDef) and n.name == '_set_laps_deleted_from_rcm')
    nodes = list(ast.walk(method))
    strings = {n.value for n in nodes if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    iterates_messages = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                           and n.func.attr == 'iterrows' and isinstance(n.func.value, ast.Attribute)
                           and n.func.value.attr == '_race_control_messages' for n in nodes)
    return dict(method=method.name, first_line=method.lineno, last_line=method.end_lineno,
                message_attribute_guard='_race_control_messages' in strings,
                iterates_race_control_messages=iterates_messages,
                reads_message_text=any(isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)
                                       and n.slice.value == 'Message' for n in nodes),
                parses_deleted_regex=any(' DELETED - ' in s for s in strings),
                parses_reinstatement_regex=any('REINSTATED' in s for s in strings),
                writes_deleted_fields={'Deleted', 'DeletedReason'} <= strings)


def main():
    package = Path(importlib.util.find_spec('fastf1').origin).parent
    core = package / 'core.py'; api = package / '_api.py'
    source = core.read_text(); result = facts(source)
    checks = ['message_attribute_guard', 'iterates_race_control_messages', 'reads_message_text',
              'parses_deleted_regex', 'parses_reinstatement_regex', 'writes_deleted_fields']
    assert all(result[k] for k in checks)
    # Negative controls must invalidate each load-bearing text-source claim.
    assert not facts(source.replace('"Message"', '"NumericFlag"'))['reads_message_text']
    assert not facts(source.replace('_race_control_messages', '_unrelated_stream'))['iterates_race_control_messages']
    assert not facts(source.replace(' DELETED - ', ' REMOVED - '))['parses_deleted_regex']
    assert not facts(source.replace('"DeletedReason"', '"OtherReason"'))['writes_deleted_fields']
    # Separate line-based reconstruction checks the AST conclusions on the same source.
    lines = source.splitlines()[result['first_line']-1:result['last_line']]
    assert sum('self._race_control_messages.iterrows()' in line for line in lines) == 2
    assert sum('row["Message"]' in line for line in lines) == 2
    assert any('("Deleted", "IsPersonalBest", "DeletedReason")' in line for line in lines)
    assert any('return' == line.strip() for line in lines)
    evidence = dict(package_version=importlib.metadata.version('fastf1'), facts=result,
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [core, api]},
                    primary_docs='https://github.com/theoehrly/fast-f1/blob/main/docs/api_reference/timing_data.rst',
                    docs_checked_via='Context7 /theoehrly/fast-f1',
                    audit_kind='post-inspection technical eligibility check, not preregistered performance',
                    event_records_acquired=False, race_control_messages_read=False, forecast_feature_created=False,
                    complete_deleted_label_available_without_text_verified=False,
                    numeric_best_time_reversal_completeness_verified=False,
                    winner_fit=False, outer_evaluated=False, evaluated_2026=False)
    (ROOT / 'results/e272_deleted_source_audit.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print('E272 AST/independent line checks/negative controls PASS; Deleted derives from message text, no complete admissible error label qualified')


if __name__ == '__main__':
    main()
