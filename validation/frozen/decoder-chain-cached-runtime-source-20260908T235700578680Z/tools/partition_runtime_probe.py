"""Actual SDK comparison of direct and intact-partitioned two-PE binaries."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(parent, source, project, verify_stopped_memory=False):
    sys.path.insert(0, str(source / 'tools'))
    from sdk_probe import verify
    verify(parent)
    execution = json.loads((parent / 'execution.json').read_text())
    result = json.loads((parent / 'results.json').read_text())
    assert execution['success'] and result['full_direct_baseline_loader_image_equal']
    assert execution['results_sha256'] == digest(parent / 'results.json')
    assert execution['manifest_sha256'] == digest(parent / 'manifest.json')
    assert result['composition_sha256'] == digest(parent / 'composition.json')
    # The copied files themselves are frozen below, including every compiler
    # ELF, RPC table and SDK I/O artifact. No runtime compilation or ELF edit.
    out = project / 'evidence' / ('partition-runtime-' + datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    shutil.copytree(parent / 'out', out / 'out')
    shutil.copytree(parent / 'baseline/out', out / 'baseline/out')
    for path, name in [(Path(__file__), 'driver.py'), (source / 'tools/sdk_probe.py', 'executor.py'),
                       (parent / 'inputs.npz', 'inputs.npz'),
                       (parent / 'composition.json', 'source-composition.json')]:
        shutil.copy2(path, out / name)
    config = json.loads((parent / 'config.json').read_text())
    width = config.get('width', 2)
    config.update(parent=parent.name, parent_manifest_sha256=digest(parent / 'manifest.json'),
                  parent_execution_sha256=digest(parent / 'execution.json'),
                  simulator_threads=1, independent_variant_processes=True,
                  verify_stopped_memory=verify_stopped_memory,
                  timeout_seconds=180 if width == 2 else 360,
                  scope='Actual SDK direct and experimentally composed intact compiler '
                        f'ELF binaries, {width} original128x112 tied matrix tiles, four '
                        f'cases each; collective={config.get("collective", False)}. Full raw weight readback '
                        'before/after; no host weight H2D or neural arithmetic. '
                        'Not official SDK composition support, full vocabulary or model inference.')
    (out / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(
        sdk_sha256=execution['sdk_sha256'],
        files={str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root, variant=None):
    import numpy as np
    from executor import verify
    from cerebras.sdk.runtime.sdkruntimepybind import (
        SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform)
    verify(root)
    os.chdir(root)
    config = json.loads((root / 'config.json').read_text())
    width = config.get('width', 2)
    data = np.load(root / 'inputs.npz')
    started = time.monotonic()

    def stage(name):
        value = dict(stage=name, elapsed_seconds=time.monotonic()-started,
                     elapsed_scope='This process only; total budget uses executor clock',
                     process_id=os.getpid(), variant=variant or 'orchestrator',
                     utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (root / 'stage.json').write_text(json.dumps(value)+'\n')
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value)+'\n')
        print(name, flush=True)

    if variant is None:
        # The failed145950 same-process second simulator is preserved. Each
        # runtime now starts in a fresh process, strictly sequentially, under
        # one executor's aggregate resource/time guard.
        reports = []
        for label in ['composed', 'baseline']:
            command = [sys.executable, str(root / 'driver.py'), '--variant-worker',
                       str(root), '--variant', label]
            stage(label+'-subprocess')
            subprocess.run(command, check=True)
            result = json.loads((root / f'{label}-results.json').read_text())
            assert result['success']
            reports.extend(result['variants'])
        for epoch in range(len(config['cases'])):
            direct = np.load(root / f'baseline-actual-{epoch}.npz')['output']
            combined = np.load(root / f'composed-actual-{epoch}.npz')['output']
            np.testing.assert_array_equal(direct.view(np.uint32), combined.view(np.uint32))
        (root / 'results.json').write_text(json.dumps(dict(success=True, variants=reports,
            composed_outputs_bit_exact_baseline=True, independent_variant_processes=True,
            scope=config['scope']), indent=2)+'\n')
        stage('complete')
        return

    reports = []
    for label, directory in [(variant, root / 'baseline' if variant == 'baseline' else root)]:
        # Separate working directories preserve both simulator/API logs.
        os.chdir(directory)
        stage(label+'-construct')
        runner = SdkRuntime('out', get_platform(None, SimfabConfig(
            suppress_trace=True, num_threads=1, dump_core=True), SdkTarget.WSE3))
        ids = {name: runner.get_id(name) for name in ['weights', 'input', 'output', 'progress', 'timing']}

        def read(name, count):
            value = np.zeros((width, count), np.float32 if name == 'output' else np.uint32)
            runner.memcpy_d2h(value.ravel(), ids[name], 0, 0, width, 1, count, streaming=False,
                data_type=MemcpyDataType.MEMCPY_16BIT if name == 'timing' else MemcpyDataType.MEMCPY_32BIT,
                order=MemcpyOrder.ROW_MAJOR, nonblock=False)
            return value

        case_reports = []
        try:
            stage(label+'-load')
            runner.load()
            stage(label+'-run')
            runner.run()
            stage(label+'-initial-weights')
            initial = read('weights', 7168)
            np.save(root / f'{label}-initial-weights.npy', initial)
            np.testing.assert_array_equal(initial, data['weights'])
            first = None
            for epoch, name in enumerate(config['cases']):
                stage(label+'-'+name)
                runner.memcpy_h2d(ids['input'], data['inputs'][epoch].copy().ravel(), 0, 0, width, 1, 112,
                    streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                    order=MemcpyOrder.ROW_MAJOR, nonblock=False)
                runner.launch('compute', nonblock=False)
                actual = dict(output=read('output', 128), progress=read('progress', 1), timing=read('timing', 6))
                file = root / f'{label}-actual-{epoch}.npz'
                np.savez(file, **actual)
                assert np.isfinite(actual['output']).all()
                assert np.all(actual['progress'] == epoch+1)
                metrics = []
                for pe in range(width):
                    expected = data['expected'][epoch, pe]
                    error = actual['output'][pe].astype(np.float64)-expected
                    l2 = float(np.linalg.norm(error)/max(np.linalg.norm(expected), 1e-30))
                    peak = float(np.max(np.abs(error))/max(np.max(np.abs(expected)), 1e-30))
                    assert l2 <= config['gate']['relative_l2'] and peak <= config['gate']['relative_peak']
                    metrics.append(dict(pe=pe, relative_l2=l2, relative_peak=peak))
                if epoch == 0:
                    first = actual['output'].copy()
                elif name == 'repeat':
                    np.testing.assert_array_equal(actual['output'].view(np.uint32), first.view(np.uint32))
                elif name == 'zero':
                    assert np.all(actual['output'] == 0)
                case_reports.append(dict(case=name, metrics=metrics, actual_sha256=digest(file)))
            stage(label+'-final-weights')
            final = read('weights', 7168)
            np.save(root / f'{label}-final-weights.npy', final)
            np.testing.assert_array_equal(final, initial)
        finally:
            stage(label+'-stop')
            runner.stop()
        stopped_memory = None
        if config.get('verify_stopped_memory', False):
            # Official sdklayout examples read actual memory after stop using
            # global fabric coordinates. This is simulator validation only.
            stage(label+'-stopped-memory')
            assert (directory / 'out.core').is_file()
            decoded = {}
            for name, expected in [('output', actual['output']),
                                   ('progress', actual['progress']), ('weights', final)]:
                rows = []
                for pe in range(width):
                    raw = np.asarray(runner.read_symbol(4+pe, 1, name, dtype='uint8'))
                    assert raw.dtype == np.uint8 and raw.nbytes == expected[pe].nbytes
                    assert raw.tobytes() == expected[pe].tobytes(), (label, pe, name)
                    rows.append(raw.copy())
                decoded[name] = np.stack(rows)
            saved = root / f'{label}-stopped-memory.npz'
            np.savez(saved, **decoded)
            stopped_memory = dict(passed=True, actual_sha256=digest(saved),
                core_sha256=digest(directory / 'out.core'),
                core_bytes=(directory / 'out.core').stat().st_size,
                scope='After-stop SDK read_symbol only; all physical PE coordinates, final output/progress/original weights bit-exact to actual D2H. No live snapshot or hardware performance claim.')
        reports.append(dict(variant=label, cases=case_reports,
            stopped_memory=stopped_memory,
            initial_weights_sha256=digest(root / f'{label}-initial-weights.npy'),
            final_weights_sha256=digest(root / f'{label}-final-weights.npy')))
    (root / f'{variant}-results.json').write_text(json.dumps(dict(success=True,
        variants=reports, scope=config['scope']), indent=2)+'\n')
    stage(variant+'-complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--from-compiled', type=Path)
    action.add_argument('--worker', type=Path)
    action.add_argument('--variant-worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--variant', choices=['baseline', 'composed'])
    parser.add_argument('--verify-stopped-memory', action='store_true')
    args = parser.parse_args()
    if args.from_compiled:
        prepare(args.from_compiled.resolve(), args.source_root.resolve(), args.project_root.resolve(),
                args.verify_stopped_memory)
    elif args.worker:
        worker(args.worker.resolve())
    elif args.variant_worker:
        if not args.variant:
            parser.error('--variant is required for a variant worker')
        worker(args.variant_worker.resolve(), args.variant)
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        config = json.loads((args.execute / 'config.json').read_text())
        execute(args.execute.resolve(), config.get('timeout_seconds', 180), rss_limit_kib=1024*1024)
