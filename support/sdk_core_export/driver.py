"""Public core export; shutdown and snapshot validation are separate."""
import argparse
import datetime
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

SDK_SHA = 'fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d'
CASES = {
    'normal-compute': dict(timeout_seconds=120, mode=0, limit=16),
    'normal-communication': dict(timeout_seconds=120, mode=1, limit=8),
    'active-compute': dict(timeout_seconds=30, mode=0, limit=1_000_000_000),
    'active-communication': dict(timeout_seconds=30, mode=1, limit=1_000_000_000),
}
FILES = ('driver.py', 'sdk_probe.py', 'sdk_core.py', 'sdk_lifecycle.py',
         'pe.csl', 'layout.csl', 'packets.csl', 'source-provenance.json')


def prepare(source, project, case, compiled):
    sys.path.insert(0, str(source))
    from sdk_probe import sha, verify
    config = dict(CASES[case], case=case, rss_limit_kib=2*1024*1024,
                  simulator_threads=2, observation_seconds=1.0,
                  host_pid_namespace=os.readlink('/proc/self/ns/pid'),
                  allow_stop_timeout=case.startswith('active-'), explicit_core_export=True,
                  scope='Public SDK2.10.1 explicit core export qualification only. '
                        'No restoration, full-model numerical, generation or hardware acceptance.')
    if compiled is None:
        raise ValueError('An accepted compile bundle is required')
    verify(compiled)
    receipt = json.loads((compiled/'execution.json').read_text())
    result = json.loads((compiled/'results.json').read_text())
    assert receipt['success'] and result['success'] and result['compile_only']
    assert receipt['manifest_sha256'] == sha(compiled/'manifest.json')
    assert receipt['results_sha256'] == sha(compiled/'results.json')
    assert receipt['sdk_sha256'] == SDK_SHA
    # Host driver/provenance change; device and SDK helpers stay exact.
    for name in FILES:
        if name not in ('driver.py', 'source-provenance.json'):
            assert sha(source/name) == sha(compiled/name), name
    assert sha(compiled/'manifest.json') == 'd19bd241da4aab2180525dea9b25d99d9a67863ec9e8e049451e6bb1f2301f3d'
    config['reused_compile_manifest_sha256'] = sha(compiled/'manifest.json')
    config['new_driver_sha256'] = sha(source/'driver.py')
    config['compile_driver_sha256'] = sha(compiled/'driver.py')
    config['compile_execution_sha256'] = sha(compiled/'execution.json')
    for name, digest in result['artifacts'].items():
        assert sha(compiled/'out'/name) == digest
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root = project/'evidence'/f'sdk-core-export-{case}-{stamp}'
    root.mkdir()
    for name in FILES:
        shutil.copy2(source/name, root/name)
    shutil.copytree(compiled/'out', root/'out')
    shutil.copy2(compiled/'execution.json', root/'compile-execution.json')
    shutil.copy2(compiled/'manifest.json', root/'compile-manifest.json')
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
    assert config['explicit_core_export'] and config['new_driver_sha256'] == sha(root/'driver.py')
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
            stage('bounded-observation')
            duration = config['observation_seconds']
            deadline = time.monotonic()+duration
            while time.monotonic() < deadline:
                time.sleep(min(.05, max(0, deadline-time.monotonic())))
            stage('observation-window-ended')
        # Public owner-thread call, before stop. Never reuse automatic out.core.
        stage('explicit-core-export')
        assert not (directory/'explicit.core').exists()
        runner.dump_elf_core('explicit.core')
        stage('explicit-core-export-returned')
        assert (directory/'explicit.core').is_file()
        with (directory/'explicit.core').open('rb') as stream:
            header = stream.read(18)
        assert header[:6] == b'\x7fELF\x02\x01' and header[16:18] == b'\x04\x00'
        exported = dict(api='SdkRuntime.dump_elf_core', returned=True,
            file='explicit.core', elf_header_hex=header.hex(), bytes=(directory/'explicit.core').stat().st_size,
            sha256=sha(directory/'explicit.core'), events=list(events),
            scope='Export receipt only; inspect exact file after simulator processes exit.')
        (directory/'core-export.json').write_text(json.dumps(exported, indent=2)+'\n')
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
    # Numerical parsing is a separate postmortem after this worker exits.
    (root/'results.json').write_text(json.dumps(dict(success=True,
        export_returned=True, stop_returned=True, numerical_validated=False,
        output_sha256={name: sha(directory/name) for name in
                      (['explicit.core', 'core-export.json', 'lifecycle.json', 'setup.npy'] +
                       (['normal.npz'] if normal is not None else []))},
        scope=config['scope']), indent=2)+'\n')
    stage('complete')


def inspect_export(root, destination):
    """Inspect a returned explicit.core after all guarded worker PIDs exit."""
    assert not destination.exists() and not destination.is_relative_to(root)
    import numpy as np
    from sdk_probe import sha, verify, process_identity
    from sdk_core import StoppedCore
    verify(root)
    execution = json.loads((root/'execution.json').read_text())
    assert execution['manifest_sha256'] == sha(root/'manifest.json')
    assert execution['sdk_sha256'] == SDK_SHA
    config = json.loads((root/'config.json').read_text())
    # Offline invocation must retain the host PID namespace (no singularity -C).
    assert os.readlink('/proc/self/ns/pid') == config['host_pid_namespace']
    assert execution['observed_process_identities']
    for pid, started in execution['observed_process_identities'].items():
        current = process_identity(int(pid))
        assert current is None or (started is not None and current != started), pid
    for process in Path('/proc').glob('[0-9]*'):
        try:
            args = (process/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b'--worker' in args or b'--execute' in args:
            assert str(root).encode() not in args, 'Input bundle still executing'
    assert execution['samples_sha256'] == sha(root/'resource-samples.json')
    assert config['explicit_core_export'] and config['new_driver_sha256'] == sha(root/'driver.py')
    assert config['case'] in CASES and all(config[k] == v for k, v in CASES[config['case']].items())
    assert config['reused_compile_manifest_sha256'] == sha(root/'compile-manifest.json')
    assert config['compile_execution_sha256'] == sha(root/'compile-execution.json')
    compile_receipt = json.loads((root/'compile-execution.json').read_text())
    assert compile_receipt['success'] and compile_receipt['manifest_sha256'] == sha(root/'compile-manifest.json')
    directory = root/'runtime'
    exported = json.loads((directory/'core-export.json').read_text())
    assert exported['returned'] and exported['api'] == 'SdkRuntime.dump_elf_core'
    assert exported['file'] == 'explicit.core'
    assert exported['bytes'] == (directory/'explicit.core').stat().st_size
    assert exported['sha256'] == sha(directory/'explicit.core')
    assert exported['events'][-1]['stage'] == 'explicit-core-export-returned'
    artifacts = json.loads((root/'compiled-files.json').read_text())
    for name, digest in artifacts.items():
        assert sha(root/'out'/name) == digest == sha(directory/'out'/name)
    names = dict(control=2, status=8, value=1, wire=2, last_payload=31)
    control = np.tile(np.array([config['mode'], config['limit']], np.uint32), 2)
    expected = np.zeros((2,8), np.uint32)
    expected[:, :2] = np.array([config['mode'],config['limit']], np.uint32)
    np.testing.assert_array_equal(np.load(directory/'setup.npy'), expected)
    normal = None
    if config['case'].startswith('normal-'):
        assert execution['success']
        assert execution['results_sha256'] == sha(root/'results.json')
        result = json.loads((root/'results.json').read_text())
        for name, digest in result['output_sha256'].items():
            assert sha(directory/name) == digest
        lifecycle = json.loads((directory/'lifecycle.json').read_text())
        assert lifecycle['primary_error'] is None and lifecycle['cleanup_errors'] == []
        with np.load(directory/'normal.npz') as data:
            assert set(data.files) == set(names)
            normal = {name:data[name].copy() for name in names}
    core = StoppedCore(directory/'explicit.core', directory/'out/bin')
    actual, bindings = core_arrays(core, names)
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
    # Snapshot checks never turn a timeout into successful shutdown.
    destination.mkdir(parents=True, exist_ok=False)
    np.savez(destination/'explicit-arrays.npz', **actual)
    (destination/'symbol-bindings.json').write_text(json.dumps(bindings, indent=2)+'\n')
    report = dict(snapshot_arrays_passed=True, diagnostic_only=True,
        execution_success=execution['success'], checkpoint_restore_qualified=False,
        actual_execution_sha256=sha(root/'execution.json'),
        explicit_core_sha256=sha(directory/'explicit.core'),
        export_receipt_sha256=sha(directory/'core-export.json'),
        manifest_sha256=sha(root/'manifest.json'),
        device_status=status.tolist(), device_value=actual['value'].tolist(),
        software_operation_flags=actual['wire'].tolist(),
        normal_core_bit_exact=normal is not None, checked_bytes=352,
        output_sha256={n:sha(destination/n) for n in ('explicit-arrays.npz','symbol-bindings.json')},
        scope='Per-PE explicit snapshot arrays only; no global atomic snapshot, restoration or full-model claim.')
    (destination/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', choices=CASES)
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    action.add_argument('--inspect', type=Path)
    parser.add_argument('--report-directory', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--project-root', type=Path)
    parser.add_argument('--compiled', type=Path)
    args = parser.parse_args()
    if args.prepare:
        if args.source_root is None or args.project_root is None:
            parser.error('--source-root and --project-root required')
        prepare(args.source_root.resolve(), args.project_root.resolve(), args.prepare,
                args.compiled.resolve() if args.compiled else None)
    elif args.inspect:
        if args.report_directory is None:
            parser.error('--report-directory required')
        inspect_export(args.inspect.resolve(), args.report_directory.resolve())
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
