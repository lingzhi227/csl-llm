"""Full-model normal readback contracts; never select or supply device tokens.

Reference winners must be frozen before SDK execution. These expectations are
an independent check of device sequence metadata, not a host inference loop.
Physical KV outside the current logical prefix may retain data after reset.
"""
import numpy as np


def sequence_expectations(prompt, limit, winners):
    prompt, winners = tuple(prompt), tuple(winners)
    if (not prompt or type(limit) is not int or not 0 <= limit <= 256 or
            len(prompt)+limit > 2048 or any(type(t) is not int or not 0 <= t < 151936
                                          for t in prompt+winners)):
        raise ValueError('Invalid pinned model request or reference token IDs')
    if len(winners) > limit or (limit == 0 and winners):
        raise ValueError('Reference exceeds requested generation')
    if limit and (not winners or (len(winners) != limit and winners[-1] not in (151643, 151645))):
        raise ValueError('Reference must reach EOS or the generation limit')
    if any(t in (151643, 151645) for t in winners[:-1]):
        raise ValueError('Reference continues after EOS')
    steps = []
    count = len(prompt) if not limit else len(prompt)+len(winners)-1
    for position in range(count):
        head = bool(limit and position+1 >= len(prompt))
        generated = max(0, position+2-len(prompt)) if head else 0
        visible = winners[:generated]
        reason = (2 if head and visible[-1] in (151643, 151645) else
                  1 if head and generated == limit else
                  3 if not limit and position+1 == len(prompt) else 0)
        steps.append(dict(position=position,
            token=prompt[position] if position < len(prompt) else winners[position-len(prompt)],
            head=head, consumed=position+1, generated=list(visible), reason=reason,
            finished=bool(reason)))
    return steps


def compare(actual, expected, gate, *, all_required=False):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not actual.size:
        raise ValueError('Numerical output shape differs from the frozen reference')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Nonfinite model output/reference')
    delta = actual.astype(np.float64)-expected.astype(np.float64)
    absolute = float(np.max(np.abs(delta)))
    norm = float(np.linalg.norm(delta)/max(float(np.linalg.norm(expected.astype(np.float64))), 1e-30))
    peak = absolute/max(float(np.max(np.abs(expected))), 1e-30)
    limits = gate if all_required else gate['both']
    normwise = norm <= limits['relative_l2'] and peak <= limits['relative_peak']
    passed = absolute <= gate['max_abs'] and normwise if all_required else absolute <= gate['max_abs'] or normwise
    if not passed:
        raise ValueError(f'Frozen numerical gate failed: {absolute}, {norm}, {peak}')
    return dict(max_abs=absolute, relative_l2=norm, relative_peak=peak)


def validate_step(actual, reference, step, *, invocation, prompt, limit, precision):
    """Validate one completed step, including every PE and all 3,168 KV stripes.

    KV status is indexed [layer, KV head, physical stripe, seen/valid], not
    serpentine collective ordinal. Logits are in original vocabulary order.
    Caller binds the normal arrays to the final actual core separately.
    """
    if type(invocation) is not int or not 1 <= invocation <= 65535:
        raise ValueError('Invocation must fit the observed 16-bit wire epoch')
    consumed = step['consumed']
    position = step['position']
    prompt, generated = tuple(prompt), tuple(step['generated'])
    if (not prompt or type(limit) is not int or not 0 <= limit <= 256 or
            len(prompt)+limit > 2048 or any(type(t) is not int or not 0 <= t < 151936
                                          for t in prompt+generated)):
        raise ValueError('Invalid request or generated token metadata')
    head = bool(limit and position+1 >= len(prompt))
    if (type(position) is not int or not 0 <= position < 2048 or consumed != position+1 or
            invocation < consumed or step['head'] != head or len(generated) > limit or
            len(generated) != (position+2-len(prompt) if head else 0) or
            (not limit and consumed > len(prompt))):
        raise ValueError('Inconsistent frozen step metadata')
    if any(t in (151643, 151645) for t in generated[:-1]):
        raise ValueError('Device trajectory continues after EOS')
    reason = (2 if head and generated[-1] in (151643, 151645) else
              1 if head and len(generated) == limit else
              3 if not limit and consumed == len(prompt) else 0)
    token = prompt[position] if position < len(prompt) else generated[-2]
    if step['reason'] != reason or step['finished'] != bool(reason) or step['token'] != token:
        raise ValueError('Frozen completion or consumed token differs from sequence semantics')
    for name, value in [('ready', invocation*2-1), ('progress', invocation*2)]:
        np.testing.assert_array_equal(actual[name], np.full((210, 193), value, np.uint32))
    np.testing.assert_array_equal(actual['control'], [len(prompt), limit])
    np.testing.assert_array_equal(actual['prompt'], prompt)
    np.testing.assert_array_equal(actual['generated'], step['generated'])
    np.testing.assert_array_equal(actual['summary'], [consumed, len(step['generated']),
        int(step['finished']), step['reason'], invocation, 0, int(step['head'])])
    np.testing.assert_array_equal(actual['controller_status'], np.tile([consumed, 0], (24, 1)))
    status = np.zeros((24, 2, 66, 2), np.uint32)
    status[..., 0] = consumed
    status[..., 1] = np.clip(consumed-32*np.arange(66), 0, 32)
    np.testing.assert_array_equal(actual['kv_status'], status)
    outputs = np.asarray(actual['layer_outputs'])
    expected = np.asarray(reference['layer_outputs'])
    if outputs.shape != (24, 896) or expected.shape != outputs.shape:
        raise ValueError('All 24 layer outputs must be observed independently')
    checks = {f'layer_{layer}': compare(outputs[layer], expected[layer],
        precision['stage_gate']['absolute_or_normwise']) for layer in range(24)}
    if step['head']:
        if np.asarray(actual['final_norm']).shape != (896,) or np.asarray(reference['final_norm']).shape != (896,):
            raise ValueError('Complete final RMS output is required')
        if np.asarray(actual['logits']).shape != (151936,) or np.asarray(reference['logits']).shape != (151936,):
            raise ValueError('Full original vocabulary is required')
        checks['logits'] = compare(actual['logits'], reference['logits'],
            precision['logit_gate']['all_required'], all_required=True)
        winner = int(np.argmax(actual['logits']))
        expected_winner = int(np.argmax(reference['logits']))
        np.testing.assert_array_equal(actual['winner'], [winner])
        if winner != expected_winner or winner != step['generated'][-1]:
            raise ValueError('Device winner differs from frozen greedy reference')
        np.testing.assert_array_equal(actual['best'], np.asarray(actual['logits'])[winner:winner+1])
        checks['final_norm'] = compare(actual['final_norm'], reference['final_norm'],
            precision['stage_gate']['absolute_or_normwise'])
    else:
        # The model controller retains the last layer output before final RMS.
        if np.asarray(actual['model_input']).tobytes() != outputs[-1].tobytes():
            raise ValueError('Last-layer stream differs from model-controller input')
    timings = [np.asarray(actual['model_timing']), *np.asarray(actual['controller_timing'])]
    if len(timings) != 25:
        raise ValueError('Model and all 24 controller timestamps are required')
    cycles = []
    for timing in timings:
        if timing.dtype != np.uint32 or timing.shape != (6,) or np.any(timing >= 65536):
            raise ValueError('SDK 16-bit timestamps must fit their uint32 transfer words')
        ticks = [sum(int(timing[k+i]) << (16*i) for i in range(3)) for k in (0, 3)]
        elapsed = (ticks[1]-ticks[0]) & ((1 << 48)-1)
        if not 0 < elapsed < 1 << 40:
            raise ValueError('Invalid controller clock interval')
        cycles.append(elapsed)
    return dict(passed=True, step=step, invocation=invocation, checks=checks,
        model_cycles=cycles[0], controller_cycles=cycles[1:],
        cycle_scope='Controller intervals contain upstream wait; do not sum them or equate simulator wall time with hardware performance.',
        scope='Normal full-model readbacks only; actual core, original weights, protocol drainage and resource acceptance remain separate.')
