"""Persistent diagnostic request scheduling over a caller-owned SDK instance.

References must already be hash-bound OfficialModelReference objects. This
module neither constructs references nor owns SDK load/run/stop. The caller
must preserve each readback, stop on failure, and validate the final actual
core against the returned request history before claiming runtime acceptance.
No generated token, weight or activation is written by this scheduler.
"""
import copy

from model_step_validate import sequence_expectations, validate_step


def run_requests(io, references, *, precision, record):
    """Run complete frozen requests, retaining physical-cache reset history.

    ``record(request_index, step_index, actual, report)`` must durably preserve
    evidence before acceptance permits the next launch. Exceptions poison the
    instance; only the caller may stop/dispose it. Normal validation does not
    establish final core, initializer, protocol or capacity acceptance.
    """
    references = tuple(references)
    if not references or io.invocation != 0 or io.prompt is not None or io.pending or io.failed:
        raise ValueError('A fresh initialized SDK instance and frozen requests are required')
    plans = []
    for reference in references:
        prompt, limit = tuple(reference.prompt), reference.limit
        steps = sequence_expectations(prompt, limit, tuple(reference.winners))
        if reference.steps != steps:
            raise ValueError('Reference schedule differs from independent sequence semantics')
        plans.append((prompt, limit, copy.deepcopy(steps)))
    if sum(len(steps) for _, _, steps in plans) > 65535:
        raise ValueError('Combined requests exceed the nonwrapping wire epoch')
    history, summaries = [], []
    actual = expected = None
    try:
        for request_index, (reference, (prompt, limit, steps)) in enumerate(zip(references, plans)):
            io.reset_request(prompt, limit)
            for step_index, plan in enumerate(steps):
                # Always read the independent original-input trace; actual
                # tokens and activations never condition this reference.
                expected = reference.step(step_index)
                actual = io.step(head=plan['head'], generated_count=len(plan['generated']))
                report = validate_step(actual, expected, plan, invocation=io.invocation,
                    prompt=prompt, limit=limit, precision=precision)
                record(request_index, step_index, actual, report)
                io.accept_step()
                if io.finished != plan['finished']:
                    raise ValueError('Accepted device completion differs from frozen request')
            # The final prefix includes every physical write in this request.
            # Earlier request prefixes survive reset outside the newer prefix.
            history.append({name: expected[name].copy() for name in ('cache_key', 'cache_value')})
            summaries.append(dict(request_index=request_index, prompt=list(prompt), limit=limit,
                steps=len(steps), invocation=io.invocation, generated=list(steps[-1]['generated']),
                reason=steps[-1]['reason']))
        # Only the final reference needs layer/logit diagnostics. Copy arrays
        # before the owner closes any NPZ-backed reference objects.
        history[-1].update({name: value.copy() for name, value in expected.items()})
    except BaseException:
        io.failed = True
        raise
    return dict(normal=actual, reference_history=history, requests=summaries,
                invocation=io.invocation, runtime_accepted=False)
