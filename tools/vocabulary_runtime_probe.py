"""Full original vocabulary runtime from audited intact compiler partitions.

Each case uses a fresh simulator process, performs device lookup and full LM
head, returns the device winner through ordinary SDK D2H, then stops. Complete
logits and all original weights are verified from the actual stopped core.
This simulator validation accommodation does not claim hardware read bandwidth
or persistent autoregressive/model inference. No neural arithmetic on the host.
"""
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
WIDTH, HEIGHT, TILES, VOCAB = 193, 50, 9496, 151936


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_compiled_receipt(directory, receipt):
    actual = {str(p.relative_to(directory)): digest(p)
              for p in directory.rglob('*') if p.is_file()}
    if actual != receipt:
        raise ValueError('Compiled output files differ from completed assembly receipt')


def prepare(assembled, fixture, source, project):
    sys.path.insert(0, str(source / 'tools'))
    from partition_executor import verify
    verify(assembled)
    original = verify(fixture)
    execution = json.loads((assembled / 'execution.json').read_text())
    result = json.loads((assembled / 'results.json').read_text())
    assert execution['success'] and result['success']
    assert execution['sdk_sha256'] == original['sdk_sha256']
    assert execution['manifest_sha256'] == digest(assembled / 'manifest.json')
    assert execution['results_sha256'] == digest(assembled / 'results.json')
    assert result['original_tiles'] == TILES and result['application_coordinates'] == WIDTH*HEIGHT
    assert result['composition_sha256'] == digest(assembled / 'composition.json')
    assert result['symbols_sha256'] == digest(assembled / 'symbols.json')
    symbols = json.loads((assembled / 'symbols.json').read_text())
    assert symbols['passed'] and symbols['application_coordinates'] == WIDTH*HEIGHT
    assert result['source_packed_sha256'] == original['files']['weights.u32.bin']
    composition = json.loads((assembled / 'composition.json').read_text())
    assert composition['passed'] and composition['coordinate_count'] == WIDTH*HEIGHT
    # Generated artifacts are not part of the assembly's input manifest.
    # Bind every copied ELF/RPC/I/O/DSR file to its completed output receipt.
    verify_compiled_receipt(assembled / 'out', composition['artifact_sha256'])
    out = project / 'evidence' / ('vocabulary-runtime-' + datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    shutil.copytree(assembled / 'out', out / 'out')
    (out / 'compiled-files.json').write_text(json.dumps(
        {str(p.relative_to(out / 'out')): digest(p) for p in (out / 'out').rglob('*') if p.is_file()},
        indent=2)+'\n')
    for name in ['inputs.npz', 'identity.npy', 'weights.u32.bin', 'precision.json']:
        shutil.copy2(fixture / name, out / name)
    for path, name in [(Path(__file__), 'driver.py'), (source / 'tools/partition_executor.py', 'executor.py'),
                       (source / 'tools/vocabulary_sdk_core.py', 'sdk_core.py'),
                       (assembled / 'symbols.json', 'source-symbols.json'),
                       (assembled / 'composition.json', 'source-composition.json')]:
        shutil.copy2(path, out / name)
    config = json.loads((fixture / 'config.json').read_text())
    config.update(assembled=assembled.name, assembled_execution_sha256=digest(assembled / 'execution.json'),
        fixture=fixture.name, fixture_manifest_sha256=digest(fixture / 'manifest.json'),
        initialization='intact-original-static-ELFs', cases=3, simulator_threads=8,
        timeout_seconds=10800, rss_limit_kib=12*1024*1024,
        gate=dict(max_abs=.01, relative_l2=2e-4, relative_peak=5e-4),
        scope='Complete original151936x896 tied lookup/head in three fresh sequential SDK processes. '
              'Each performs lookup then head, normal device winner D2H and stopped-core full logits, '
              'all original weights, identity, progress and control. Repeat is across fresh processes; '
              'not persistent repeat/reset, final RMS, 24 layers, generation or hardware performance.')
    (out / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(sdk_sha256=original['sdk_sha256'],
        files={str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root, case=None):
    import numpy as np
    from executor import verify
    verify(root)
    config = json.loads((root / 'config.json').read_text())
    data = np.load(root / 'inputs.npz')
    start = time.monotonic()

    def stage(name):
        record = dict(stage=name, case=case, process_id=os.getpid(),
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            process_elapsed_seconds=time.monotonic()-start)
        (root / 'stage.json').write_text(json.dumps(record)+'\n')
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print(name, flush=True)

    if case is None:
        cases = []
        for index in range(config['cases']):
            stage(f'child-{index}')
            subprocess.run([sys.executable, str(root / 'driver.py'), '--case-worker',
                            str(root), '--case', str(index)], check=True)
            record = json.loads((root / f'case-{index}/results.json').read_text())
            assert record['success']
            cases.append(dict(record, results_sha256=digest(root / f'case-{index}/results.json')))
        first = np.load(root / 'case-0/actual.npz')['logits']
        repeated = np.load(root / 'case-2/actual.npz')['logits']
        np.testing.assert_array_equal(first.view(np.uint32), repeated.view(np.uint32))
        (root / 'results.json').write_text(json.dumps(dict(success=True, cases=cases,
            fresh_process_repeat_bit_exact=True, scope=config['scope']), indent=2)+'\n')
        stage('complete')
        return

    from cerebras.sdk.runtime.sdkruntimepybind import (
        SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform)
    directory = root / f'case-{case}'
    directory.mkdir()
    # SDK-generated loader artifacts belong to this case alone. Never share a
    # writable output directory across independent runtime instances.
    shutil.copytree(root / 'out', directory / 'out')
    compiled_files = json.loads((root / 'compiled-files.json').read_text())

    def verify_case_images():
        for name, expected in compiled_files.items():
            assert digest(directory / 'out' / name) == expected, name

    verify_case_images()
    os.chdir(directory)
    runner = SdkRuntime('out', get_platform(None, SimfabConfig(suppress_trace=True,
        num_threads=config['simulator_threads'], dump_core=True), SdkTarget.WSE3))
    names = ['identity', 'control', 'input', 'progress', 'winner', 'best']
    ids = {name: runner.get_id(name) for name in names}

    def transfer(name, x, y, width, height, count, value=None):
        stage(f'{"d2h" if value is None else "h2d"}-{name}')
        options = dict(streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
            order=MemcpyOrder.ROW_MAJOR, nonblock=False)
        if value is not None:
            runner.memcpy_h2d(ids[name], np.ascontiguousarray(value).ravel(),
                              x, y, width, height, count, **options)
            return None
        value = np.zeros(width*height*count, np.float32 if name == 'best' else np.uint32)
        runner.memcpy_d2h(value, ids[name], x, y, width, height, count, **options)
        return value

    identities = np.load(root / 'identity.npy')
    assert identities.shape == (HEIGHT, WIDTH) and identities.dtype == np.dtype('<u4')
    normal = {}
    try:
        stage('load'); runner.load()
        stage('run'); runner.run()
        runner.launch('initialize', nonblock=False)
        transfer('identity', 0, 0, WIDTH, HEIGHT, 1, identities)
        for mode in [0, 1]:
            metadata = np.tile(np.array([mode, data['token_ids'][case]], np.uint32), WIDTH*HEIGHT)
            transfer('control', 0, 0, WIDTH, HEIGHT, 2, metadata)
            if mode:
                transfer('input', 192, 0, 1, 1, 896, data['inputs'][case])
            stage(f'prepare-{mode}'); runner.launch('prepare', nonblock=False)
            ready = transfer('progress', 192, 0, 1, 1, 1)
            assert int(ready[0]) == mode*2+1
            stage(f'compute-{mode}'); runner.launch('compute', nonblock=False)
            done = transfer('progress', 192, 0, 1, 1, 1)
            assert int(done[0]) == mode*2+2
            normal[f'controller_progress_{mode}'] = done
        normal['winner'] = transfer('winner', 192, 0, 1, 1, 1)
        normal['best'] = transfer('best', 192, 0, 1, 1, 1)
        np.savez(directory / 'normal-d2h.npz', **normal)
    finally:
        stage('stop'); runner.stop()

    from sdk_core import StoppedCore
    stage('actual-core-read')
    verify(root)
    verify_case_images()
    (directory / 'runtime-artifacts.json').write_text(json.dumps(
        {str(p.relative_to(directory)): digest(p) for p in (directory / 'out').rglob('*') if p.is_file()},
        indent=2)+'\n')
    core = StoppedCore(directory / 'out.core', directory / 'out/bin')
    assert set(core.symbols) == {(x+4, y+1) for y in range(HEIGHT) for x in range(WIDTH)}

    def read(x, y, name, dtype, count):
        dtype = np.dtype(dtype)
        return np.frombuffer(core.read(x+4, y+1, name, dtype.itemsize*count), dtype=dtype).copy()

    progress = np.empty((HEIGHT, WIDTH), np.uint32)
    with (root / 'weights.u32.bin').open('rb') as packed:
        for y in range(HEIGHT):
            for x in range(WIDTH):
                progress[y, x] = read(x, y, 'progress', '<u4', 1)[0]
                assert read(x, y, 'identity', '<u4', 1)[0] == identities[y, x]
                assert read(x, y, 'control', '<u4', 2).tolist() == [1, int(data['token_ids'][case])]
                if x < 192 and y*192+x < TILES:
                    expected = packed.read(28672)
                    assert len(expected) == 28672
                    assert core.read(x+4, y+1, 'weights', 28672) == expected, (x, y)
        assert not packed.read(1)
    assert np.all(progress == 4)
    logits = np.concatenate([read((group % 24)*8, group//24, 'result', '<f4', 128)
                             for group in range(1187)])
    lookup = read(192, 0, 'lookup', '<f4', 896)
    np.testing.assert_array_equal(lookup.view(np.uint32), data['lookup'][case].view(np.uint32))
    assert np.isfinite(logits).all() and logits.shape == (VOCAB,)
    for name, dtype in [('winner', '<u4'), ('best', '<f4')]:
        assert read(192, 0, name, dtype, 1).tobytes() == normal[name].tobytes()
    winner = int(normal['winner'][0])
    assert winner == int(np.argmax(logits)) and normal['best'][0] == logits[winner]
    metrics = {}
    for name in ['expected', 'official']:
        reference = data[name][case].astype(np.float64)
        error = logits.astype(np.float64)-reference
        metric = dict(max_abs=float(np.max(np.abs(error))),
            relative_l2=float(np.linalg.norm(error)/max(np.linalg.norm(reference), 1e-30)),
            relative_peak=float(np.max(np.abs(error))/max(np.max(np.abs(reference)), 1e-30)))
        assert all(metric[key] <= config['gate'][key] for key in metric), metric
        assert winner == int(np.argmax(reference))
        metrics[name] = metric
    if case == 1:
        assert np.all(logits == 0) and winner == 0
    timing = read(192, 0, 'timing', '<u2', 6)
    ticks = [sum(int(timing[k+i]) << (16*i) for i in range(3)) for k in [0, 3]]
    cycles = (ticks[1]-ticks[0]) & ((1 << 48)-1)
    assert 0 < cycles < 1 << 40
    np.savez(directory / 'actual.npz', logits=logits, lookup=lookup, progress=progress, timing=timing)
    (directory / 'results.json').write_text(json.dumps(dict(success=True, case=case,
        runtime_manifest_sha256=digest(root / 'manifest.json'),
        compiled_files_sha256=digest(root / 'compiled-files.json'),
        source_composition_sha256=digest(root / 'source-composition.json'),
        source_symbols_sha256=digest(root / 'source-symbols.json'),
        runtime_artifacts_sha256=digest(directory / 'runtime-artifacts.json'),
        actual_sha256=digest(directory / 'actual.npz'), normal_d2h_sha256=digest(directory / 'normal-d2h.npz'),
        core_sha256=digest(directory / 'out.core'), core_bytes=(directory / 'out.core').stat().st_size,
        original_weights_sha256=digest(root / 'weights.u32.bin'), all9496_original_weights_exact=True,
        all9650_identity_control_progress_exact=True, winner=winner, metrics=metrics,
        controller_cycles=cycles, scope=config['scope']), indent=2)+'\n')
    stage('case-complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--from-assembled', type=Path)
    action.add_argument('--worker', type=Path)
    action.add_argument('--case-worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--case', type=int, choices=range(3))
    args = parser.parse_args()
    if args.from_assembled:
        if not args.fixture:
            parser.error('--fixture required')
        prepare(args.from_assembled.resolve(), args.fixture.resolve(), args.source_root.resolve(), args.project_root.resolve())
    elif args.worker:
        worker(args.worker.resolve())
    elif args.case_worker:
        if args.case is None:
            parser.error('--case required')
        worker(args.case_worker.resolve(), args.case)
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        config = json.loads((args.execute / 'config.json').read_text())
        execute(args.execute.resolve(), config['timeout_seconds'], rss_limit_kib=config['rss_limit_kib'])
