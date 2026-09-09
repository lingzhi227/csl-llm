"""Persistent fixed-corpus runtime body for a future guarded SDK driver.

The owner must admit the complete original assembly and successful first-model
integration, freeze dependencies/references/budget, and verify the stopped core.
This body cannot create an executable bundle or qualify S5/S6 by itself.
"""
import gc
import json
from pathlib import Path

import numpy as np

from model_fixed_requests import open_references, REQUEST_IDS
from model_request_sequence import run_requests
from model_runtime_io import ModelRuntimeIO
from sdk_probe import sha
from sdk_lifecycle import stop_and_record


def run(runner, *, directory, reference_directory, precision,
        memcpy_data_type, memcpy_order, stage):
    """Own load/run/initialize/stop of one already configured SDK instance.

    The runner's core output must be configured in ``directory`` by its owner.
    Save each partial readback and validated step separately. A stop failure
    still preserves an evidence receipt; no runtime-success file is emitted.
    """
    directory = Path(directory).resolve()
    if not directory.is_dir() or {p.name for p in directory.iterdir()} != {'out'} or not (directory/'out').is_dir():
        raise ValueError('Require a fresh runtime directory containing only the verified out artifact')
    references = []
    io = None
    current = [0, 0]
    first_request = {}
    files = set()
    primary_error = None

    def step_path(name):
        request, step = current
        path = directory/f'request-{request:02d}'/f'step-{step:04d}'
        path.mkdir(parents=True, exist_ok=True)
        files.add(path/name)
        return path/name

    def save(actual):
        np.savez(step_path('normal-d2h.npz'), **actual)

    def record(request, step, actual, report):
        if current != [request, step]:
            raise ValueError('Readback evidence path differs from the frozen schedule')
        # Repeat the original arithmetic request after two distinct resets.
        # Epochs and timing intentionally differ; numerical output bits must not.
        names = ('layer_outputs', 'model_input', 'generated') + (('logits', 'winner', 'best') if 'logits' in actual else ())
        if request == 0:
            first_request[step] = {name: actual[name].copy() for name in names}
        elif request == len(REQUEST_IDS)-1:
            original = first_request[step]
            if set(original) != set(names) or any(actual[name].dtype != value.dtype or
                    actual[name].shape != value.shape or actual[name].tobytes() != value.tobytes()
                    for name, value in original.items()):
                raise ValueError('Reset-repeat numerical output is not bit-exact')
        step_path('normal-report.json').write_text(json.dumps(report, indent=2)+'\n')
        current[1] += 1
        if current[1] == len(references[request].steps):
            current[:] = [request+1, 0]

    try:
        references = open_references(reference_directory)
        stage('load'); runner.load()
        stage('run'); runner.run()
        stage('initialize'); runner.launch('initialize', nonblock=False)
        io = ModelRuntimeIO(runner, memcpy_data_type=memcpy_data_type,
            memcpy_order=memcpy_order, save=save, stage=stage)
        result = run_requests(io, references, precision=precision, record=record)
        path = directory/'requests.json'
        path.write_text(json.dumps(dict(requests=result['requests'], invocation=result['invocation'],
            reset_repeat_bit_exact=True, runtime_accepted=False), indent=2)+'\n')
        files.add(path)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        def receipt(errors):
            if (directory/'out.core').is_file():
                files.add(directory/'out.core')
            (directory/'runtime-evidence.json').write_text(json.dumps(dict(
                files={str(p.relative_to(directory)): sha(p) for p in sorted(files) if p.is_file()},
                **errors, runtime_accepted=False), indent=2)+'\n')
        stop_and_record(runner, stage=stage, record=receipt,
                        primary_error=primary_error, references=references)
    del io
    gc.collect()
    # The caller still owns the runner reference. It must release that object
    # before loading a full-fabric core, then validate immutable bytes, final
    # normal/core outputs, physical KV history and protocol drainage.
    return result
