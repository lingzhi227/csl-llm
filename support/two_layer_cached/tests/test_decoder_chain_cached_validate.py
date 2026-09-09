"""Reject wrong cached arithmetic and stale logical state despite valid output delivery."""
import copy
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from decoder_chain_cached_validate import case_plan, validate_case, validate_repeat


def fixture():
    original = {f'layer_{layer}_output': np.zeros((1, 896), np.float32) for layer in (0, 1)}
    cached = {f'layer_{layer}_output': np.stack([np.zeros(896, np.float32),
              np.full(896, layer+1, np.float32)]) for layer in (0, 1)}
    cases = []
    for index, plan in enumerate(case_plan()):
        actual = dict(ready=np.full(2580, plan['invocation']*2-1, np.uint32),
                      progress=np.full(2580, plan['invocation']*2, np.uint32))
        for layer in (0, 1):
            source = cached if index == 1 else original
            actual[f'layer_{layer}_output'] = source[f'layer_{layer}_output'][plan['position']].copy()
            actual[f'controller_status_{layer}'] = np.array([plan['position']+1, 0], np.uint32)
            for head in (0, 1):
                status = np.zeros((66, 2), np.uint32)
                status[:, 0] = plan['position']+1; status[0, 1] = plan['position']+1
                actual[f'kv_status_{layer}_{head}'] = status
        actual['output'] = actual['layer_1_output'].copy()
        for name in ('endpoint_timing', 'layer_0_timing', 'layer_1_timing'):
            actual[name] = np.array([1, 0, 0, 2, 0, 0], np.uint32)
        cases.append(actual)
    policy = dict(max_abs=1e-5, both=dict(relative_l2=2e-4, relative_peak=5e-4))
    return cases, original, cached, policy


def test_independent_cached_output_and_complete_status_checks():
    cases, original, cached, policy = fixture()
    for index, case in enumerate(cases):
        assert validate_case(case, index, original, cached, policy)['passed']
    assert validate_repeat(cases)['passed']
    for change in (lambda c: c['layer_1_output'].fill(0),
                   lambda c: c['kv_status_1_1'].__setitem__((1, 1), 2),
                   lambda c: c['controller_status_0'].__setitem__(0, 1),
                   lambda c: c['ready'].__setitem__(100, 1)):
        bad = copy.deepcopy(cases[1]); change(bad)
        bad['output'] = bad['layer_1_output'].copy()
        with pytest.raises(AssertionError):
            validate_case(bad, 1, original, cached, policy)


def test_reset_bit_equality_is_stronger_than_numerical_gate():
    cases, original, cached, policy = fixture()
    for name in ('output', 'layer_0_output', 'layer_1_output'):
        cases[2][name][0] = 1e-7
    assert validate_case(cases[2], 2, original, cached, policy)['passed']
    with pytest.raises(AssertionError):
        validate_repeat(cases)
