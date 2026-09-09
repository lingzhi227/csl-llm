"""Reject contradictory CSL-LLM recovery checkpoints before selecting work.

This checks recorded state only. A remote process/immutable execution check is
still mandatory; a missing local execution file does not prove a job is alive.
"""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = re.compile(r'\b[a-z][a-z0-9-]*-\d{8}T\d{6,12}Z\b')


def validate(state, evidence):
    if state.get('active_project') != 'csl-llm':
        raise ValueError('The explicit current objective is CSL-LLM')
    if 'CSL-LLM' not in state.get('objective', ''):
        raise ValueError('Missing explicit CSL-LLM objective')
    buckets = {}
    for name in ['latest_accepted', 'latest_completed_pending_review',
                 'prepared_not_executed', 'latest_failed']:
        buckets[name] = {match.group() for item in state.get(name, [])
                         for match in BUNDLE.finditer(item)}
    prepared = buckets['prepared_not_executed']
    completed = buckets['latest_accepted'] | buckets['latest_completed_pending_review'] | buckets['latest_failed']
    if prepared & completed:
        raise ValueError(f'Completed bundles still marked unexecuted: {sorted(prepared & completed)}')
    contradictory = buckets['latest_accepted'] & buckets['latest_failed']
    if contradictory:
        raise ValueError(f'Failed bundles marked accepted: {sorted(contradictory)}')
    active = state.get('active_sdk')
    offline = state.get('active_offline')
    active_ids = set()
    for running in [active, offline]:
        if not running:
            continue
        bundle = running.get('bundle', '')
        if not BUNDLE.fullmatch(bundle):
            raise ValueError('Active run requires an exact frozen bundle ID')
        if bundle in active_ids:
            raise ValueError(f'Run recorded as multiple active jobs: {bundle}')
        active_ids.add(bundle)
        if bundle in prepared | completed:
            raise ValueError(f'Active bundle also marked unexecuted/completed: {bundle}')
        if (evidence / bundle / 'execution.json').exists():
            raise ValueError(f'Active bundle already has an immutable execution result: {bundle}')
    return dict(passed=True, active_bundle=active['bundle'] if active else None,
                active_offline_bundle=offline['bundle'] if offline else None,
                scope='Recorded checkpoint consistency only. Reconcile actual owned remote processes and execution files before dispatch.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    print(json.dumps(validate(json.loads((root / 'RESUME.json').read_text()),
                              root / 'evidence'), indent=2))
