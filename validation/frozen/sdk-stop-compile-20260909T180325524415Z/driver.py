"""Bounded owner-thread stop qualification, not checkpoint restoration."""
import argparse
import datetime
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SDK_SHA = 'fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d'
CASES = {
    'compile': dict(timeout_seconds=120, mode=None, limit=None),
    'normal-compute': dict(timeout_seconds=120, mode=0, limit=16),
    'normal-communication': dict(timeout_seconds=120, mode=1, limit=8),
    'active-compute': dict(timeout_seconds=120, mode=0, limit=1_000_000_000),
    'active-communication': dict(timeout_seconds=120, mode=1, limit=1_000_000_000),
    'guard': dict(timeout_seconds=15, mode=0, limit=1_000_000_000),
}
FILES = ('driver.py', 'sdk_probe.py', 'sdk_core.py', 'sdk_lifecycle.py',
         'pe.csl', 'layout.csl', 'packets.csl', 'source-provenance.json')


def prepare(source, project, case, compiled):
    sys.path.insert(0, str(source))
    from sdk_probe import sha, verify
    config = dict(CASES[case], case=case, rss_limit_kib=2*1024*1024,
                  simulator_threads=2, observation_seconds=1.0,
                  expected_outer_failure=case == 'guard',
                  scope='Small SDK2.10.1 stop/diagnostic-core qualification only. '
                        'No restoration, full-model numerical, generation or hardware acceptance.')
    if case != 'compile':
        if compiled is None:
            raise ValueError('An accepted compile bundle is required')
        verify(compiled)
        receipt = json.loads((compiled/'execution.json').read_text())
        result = json.loads((compiled/'results.json').read_text())
        assert receipt['success'] and result['success'] and result['compile_only']
        assert receipt['manifest_sha256'] == sha(compiled/'manifest.json')
        assert receipt['results_sha256'] == sha(compiled/'results.json')
        assert receipt['sdk_sha256'] == SDK_SHA
        for name in FILES:
            assert sha(source/name) == sha(compiled/name), name
        config['compile_execution_sha256'] = sha(compiled/'execution.json')
        for name, digest in result['artifacts'].items():
            assert sha(compiled/'out'/name) == digest
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root = project/'evidence'/f'sdk-stop-{case}-{stamp}'
    root.mkdir()
    for name in FILES:
        shutil.copy2(source/name, root/name)
    if case != 'compile':
        shutil.copytree(compiled/'out', root/'out')
        shutil.copy2(compiled/'execution.json', root/'compile-execution.json')
        (root/'compiled-files.json').write_text(json.dumps(result['artifacts'], indent=2)+'\n')
    (root/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (root/'manifest.json').write_text(json.dumps(dict(sdk_sha256=SDK_SHA,
        files={str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}), indent=2)+'\n')
    print(root, flush=True)


def compute_oracle(count):
    """Independent GF(2) matrix exponentiation, bounded by log2(count)."""
    import numpy as np
    assert 0 <= count <= 1_000_000_000
    identity = np.eye(32, dtype=np.uint64)
    left13 = np.zeros((32, 32), dtype=np.uint64)
    right17 = np.zeros((32, 32), dtype=np.uint64)
    left5 = np.zeros((32, 32), dtype=np.uint64)
    for bit in range(32):
        if bit >= 13:
            left13[bit, bit-13] = 1
        if bit+17 < 32:
            right17[bit, bit+17] = 1
        if bit >= 5:
            left5[bit, bit-5] = 1
    transition = ((identity+left5) @ (identity+right17) @ (identity+left13)) % 2
    vector = np.array([(7 >> bit) & 1 for bit in range(32)], dtype=np.uint64)
    while count:
        if count & 1:
            vector = (transition @ vector) % 2
        transition = (transition @ transition) % 2
        count >>= 1
    return sum(int(bit) << i for i, bit in enumerate(vector))


def core_arrays(core, names):
    import numpy as np
    arrays, bindings = {}, {}
    for label, count in names.items():
        rows = []
        for x in (4, 5):
            table = core.symbols[(x, 1)]
            candidates = {name: entry for name, entry in table.items()
                          if name == label or name.split('$$')[-1] == label}
            if not candidates or len(set(candidates.values())) != 1:
                raise ValueError(f'No unique actual compiled object for {x}:{label}: {candidates}')
            address, size = next(iter(candidates.values()))
            assert size == count*4
            selected = sorted(candidates)[0]
            rows.append(np.frombuffer(core.read(x, 1, selected, size), np.uint32).copy())
            bindings[f'{x},1:{label}'] = dict(address=address, size=size, aliases=sorted(candidates))
        arrays[label] = np.stack(rows)
    return arrays, bindings


def worker(root):
    sys.path.insert(0, str(root))
    from sdk_probe import verify, sha
    verify(root)
    config = json.loads((root/'config.json').read_text())
    case = config['case']
    assert case in CASES and all(config[k] == v for k, v in CASES[case].items())
    assert config['rss_limit_kib'] == 2*1024*1024
    assert config['simulator_threads'] == 2 and config['observation_seconds'] == 1.0
    if case == 'compile':
        os.chdir(root)
        command = ['cslc', 'layout.csl', '--arch=wse3', '--fabric-dims=9,3',
                   '--fabric-offsets=4,1', '-o=out', '--memcpy', '--channels=1',
                   '--dump-dsr-alloc-graph']
        (root/'compile-command.json').write_text(json.dumps(command)+'\n')
        subprocess.run(command, check=True)
        artifacts = {str(p.relative_to(root/'out')): sha(p)
                     for p in (root/'out').rglob('*') if p.is_file()}
        assert list((root/'out/bin').glob('*.elf'))
        (root/'results.json').write_text(json.dumps(dict(success=True, compile_only=True,
            artifacts=artifacts, scope=config['scope']), indent=2)+'\n')
        return
    import numpy as np
    from sdk_core import StoppedCore
    from sdk_lifecycle import stop_and_record
    from cerebras.sdk.runtime.sdkruntimepybind import (
        SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform)
    assert sha(root/'compile-execution.json') == config['compile_execution_sha256']
    artifacts = json.loads((root/'compiled-files.json').read_text())
    for name, digest in artifacts.items():
        assert sha(root/'out'/name) == digest
    directory = root/'runtime'
    directory.mkdir()
    # Preserve frozen input artifacts even if the SDK writes runtime side files.
    shutil.copytree(root/'out', directory/'out')
    os.chdir(directory)
    events = []
    started = time.monotonic()
    def stage(name):
        event = dict(stage=name, elapsed_seconds=time.monotonic()-started)
        events.append(event)
        (root/'stage.json').write_text(json.dumps(event)+'\n')
        with (root/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        print(name, flush=True)
    runner = SdkRuntime('out', get_platform(None,
        SimfabConfig(num_threads=2, suppress_trace=True, dump_core=True), SdkTarget.WSE3))
    options = dict(streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                   order=MemcpyOrder.ROW_MAJOR, nonblock=False)
    names = dict(control=2, status=8, value=1, wire=2, last_payload=31)
    ids = {name: runner.get_id(name) for name in names}
    normal = None
    primary = None
    def read(name):
        data = np.zeros(2*names[name], np.uint32)
        runner.memcpy_d2h(data, ids[name], 0, 0, 2, 1, names[name], **options)
        return data.reshape(2, names[name])
    try:
        stage('load'); runner.load()
        stage('run'); runner.run()
        stage('initialize'); runner.launch('initialize', nonblock=False)
        control = np.tile(np.array([config['mode'], config['limit']], np.uint32), 2)
        runner.memcpy_h2d(ids['control'], control, 0, 0, 2, 1, 2, **options)
        runner.launch('setup', nonblock=False)
        setup = read('status')
        expected = np.zeros((2, 8), np.uint32)
        expected[:, :2] = np.array([config['mode'], config['limit']], np.uint32)
        np.testing.assert_array_equal(setup, expected)
        np.save(directory/'setup.npy', setup)
        stage('launch-work'); runner.launch('work', nonblock=False)
        stage('owner-control-returned')
        if case.startswith('normal-'):
            normal = {}
            for name in names:
                stage('read-'+name)
                normal[name] = read(name)
                np.savez(directory/'normal.npz', **normal)
        else:
            stage('guard-hold' if case == 'guard' else 'bounded-observation')
            duration = 600.0 if case == 'guard' else config['observation_seconds']
            deadline = time.monotonic()+duration
            while time.monotonic() < deadline:
                time.sleep(min(.05, max(0, deadline-time.monotonic())))
            if case == 'guard':
                raise RuntimeError('Outer guard failed to enforce its predeclared deadline')
            stage('observation-window-ended')
    except BaseException as error:
        primary = error
        raise
    finally:
        def receipt(errors):
            report = dict(**errors, core_exists=(directory/'out.core').is_file(),
                          events=events, scope=config['scope'])
            (directory/'lifecycle.json').write_text(json.dumps(report, indent=2)+'\n')
        stop_and_record(runner, stage=stage, record=receipt, primary_error=primary)
    del runner
    gc.collect()
    stage('core-inspection')
    for name, digest in artifacts.items():
        assert sha(directory/'out'/name) == digest
    core = StoppedCore(directory/'out.core', directory/'out/bin')
    actual, bindings = core_arrays(core, names)
    np.savez(directory/'core-arrays.npz', **actual)
    (directory/'symbol-bindings.json').write_text(json.dumps(bindings, indent=2)+'\n')
    status = actual['status'].astype(np.uint64)
    np.testing.assert_array_equal(actual['control'], control.reshape(2, 2))
    np.testing.assert_array_equal(status[:, :2], expected[:, :2])
    assert np.all(status[:, 4] == 1) and np.all(status[:, 7] == 0)
    assert np.all(actual['wire'] <= 1)
    if normal is not None:
        for name in names:
            np.testing.assert_array_equal(actual[name], normal[name])
        assert np.all(status[:, 2] == config['limit']) and np.all(status[:, 6] == config['limit'])
        assert np.all(status[:, 3] == 1) and np.all(status[:, 5] == 0)
        assert not actual['wire'].any()
        if config['mode'] == 0:
            assert np.all(actual['value'] == compute_oracle(config['limit']))
        else:
            seq = config['limit']-1
            want = np.array([seq ^ (i*0x01010101) ^ 0x5a5a5a5a for i in range(31)], np.uint32)
            np.testing.assert_array_equal(actual['last_payload'][1], want)
    else:
        assert np.all(status[:, 3] == 0), 'Early-stop case completed before stop'
        assert np.all(status[:, 2] < config['limit'])
        if config['mode'] == 0:
            assert np.all(status[:, 6] > 0), 'No witnessed committed device work'
            assert np.all(status[:, 2] >= status[:, 6])
            assert np.all(status[:, 2]-status[:, 6] <= 1)
            for row, value in zip(status, actual['value'][:, 0]):
                assert int(value) in (compute_oracle(int(row[2])), compute_oracle(int(row[6])))
            assert not actual['wire'].any()
        else:
            sender, receiver = status[:, 2]
            assert 0 < sender <= receiver <= sender+1
            assert actual['wire'].any(), 'No witnessed outstanding packet operation at stop'
            if actual['wire'][1, 1] == 0:
                seq = int(receiver)-1
                want = np.array([seq ^ (i*0x01010101) ^ 0x5a5a5a5a for i in range(31)], np.uint32)
                np.testing.assert_array_equal(actual['last_payload'][1], want)
    lifecycle = json.loads((directory/'lifecycle.json').read_text())
    assert lifecycle['primary_error'] is None and lifecycle['cleanup_errors'] == []
    output_names = ['setup.npy', 'out.core', 'lifecycle.json', 'core-arrays.npz', 'symbol-bindings.json']
    if normal is not None:
        output_names.append('normal.npz')
    report = dict(success=True, case=case, diagnostic_only=True, checkpoint_restore_qualified=False,
        device_status=status.tolist(), observed_wire=actual['wire'].tolist(),
        stop_returned=True, core_object_bytes_checked=sum(names.values())*2*4,
        normal_core_bit_exact=normal is not None,
        output_sha256={name: sha(directory/name) for name in output_names}, scope=config['scope'])
    (root/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    stage('complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', choices=CASES)
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--project-root', type=Path)
    parser.add_argument('--compiled', type=Path)
    args = parser.parse_args()
    if args.prepare:
        if args.source_root is None or args.project_root is None:
            parser.error('--source-root and --project-root required')
        prepare(args.source_root.resolve(), args.project_root.resolve(), args.prepare,
                args.compiled.resolve() if args.compiled else None)
    elif args.worker:
        worker(args.worker.resolve())
    else:
        root = args.execute.resolve()
        sys.path.insert(0, str(root))
        from sdk_probe import execute, verify
        verify(root)
        config = json.loads((root/'config.json').read_text())
        assert config['case'] in CASES
        assert config['timeout_seconds'] == CASES[config['case']]['timeout_seconds']
        assert config['rss_limit_kib'] == 2*1024*1024
        execute(root, config['timeout_seconds'], rss_limit_kib=config['rss_limit_kib'])
