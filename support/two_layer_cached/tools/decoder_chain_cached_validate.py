"""Normal SDK readback checks for one persistent two-layer three-case run."""
import numpy as np


def case_plan():
    return [dict(label='first', reset=True, position=0, invocation=1, input_index=0),
            dict(label='cached', reset=False, position=1, invocation=2, input_index=1),
            dict(label='reset-repeat', reset=True, position=0, invocation=3, input_index=0)]


def compare(actual, expected, policy):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape == (896,)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    actual, expected = actual.astype(np.float64), expected.astype(np.float64)
    error = actual-expected
    absolute = float(np.max(np.abs(error)))
    norm = float(np.linalg.norm(error)/max(float(np.linalg.norm(expected)), 1e-30))
    peak = absolute/max(float(np.max(np.abs(expected))), 1e-30)
    assert absolute <= policy['max_abs'] or (
        norm <= policy['both']['relative_l2'] and peak <= policy['both']['relative_peak']), (absolute, norm, peak)
    return dict(max_abs=absolute, relative_l2=norm, relative_peak=peak)


def validate_case(actual, index, original, cached, policy):
    plan = case_plan()[index]
    expected = cached if index == 1 else original
    position = plan['position']
    np.testing.assert_array_equal(actual['ready'], np.full(2580, plan['invocation']*2-1, np.uint32))
    np.testing.assert_array_equal(actual['progress'], np.full(2580, plan['invocation']*2, np.uint32))
    assert actual['output'].tobytes() == actual['layer_1_output'].tobytes()
    checks, cycles = {}, {}
    for layer in (0, 1):
        checks[f'layer_{layer}_output'] = compare(actual[f'layer_{layer}_output'],
            expected[f'layer_{layer}_output'][position], policy)
        np.testing.assert_array_equal(actual[f'controller_status_{layer}'], [position+1, 0])
        status = np.zeros((66, 2), np.uint32)
        status[:, 0] = position+1
        status[0, 1] = position+1
        for head in (0, 1):
            np.testing.assert_array_equal(actual[f'kv_status_{layer}_{head}'], status)
    for name in ('endpoint_timing', 'layer_0_timing', 'layer_1_timing'):
        timing = actual[name]
        assert timing.dtype == np.uint32 and timing.shape == (6,) and np.all(timing < 65536)
        ticks = [sum(int(timing[k+i]) << (16*i) for i in range(3)) for k in (0, 3)]
        cycles[name] = (ticks[1]-ticks[0]) & ((1 << 48)-1)
        assert 0 < cycles[name] < 1 << 40
    return dict(case=plan, passed=True, checks=checks, cycles=cycles,
        cycle_scope='Endpoint covers both layers. Controller clocks include upstream wait and must not be summed.')


def validate_repeat(actual):
    assert len(actual) == 3
    for name in ('output', 'layer_0_output', 'layer_1_output'):
        assert actual[0][name].tobytes() == actual[2][name].tobytes(), name
    return dict(passed=True, reset_repeat_bit_exact=True,
                scope='Actual first and reset-repeat endpoint and both controller outputs are bit-exact.')
