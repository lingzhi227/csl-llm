"""Bounded transfer diagnosis on the complete vocabulary fabric, not inference."""
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


def diagnostic_cases(short_first=False):
    pairs = [(1, 1), (1, 8), (1, 32), (1, 128), (8, 32), (192, 32), (1, 896)] if short_first else [(1, 7168), (192, 32), (8, 896)]
    return [dict(width=width, height=1, words=words, first_tile=0, source_word_offset=0) for width, words in pairs]


def prepare(before, project, source, short_first=False):
    manifest = json.loads((before / 'manifest.json').read_text())
    config = json.loads((before / 'config.json').read_text())
    if config.get('initialization') != 'sdk' or config.get('matrix_tiles') != 9496:
        raise ValueError('Diagnostic must retain the full vocabulary fixture')
    for name, sha in manifest['files'].items():
        if Path(name).name != name or digest(before / name) != sha:
            raise ValueError(f'Changed parent input {name}')
    out = project / 'evidence' / ('vocabulary-transfer-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    for name in manifest['files']:
        shutil.copy2(before / name, out / name)
    shutil.copy2(Path(__file__), out / 'driver.py')
    shutil.copy2(source / 'tools/sdk_probe.py', out / 'executor.py')
    config.update(parent_manifest_sha256=digest(before / 'manifest.json'),
        diagnostic_cases=diagnostic_cases(short_first), short_depth_first=short_first,
        scope='Transfer-only diagnosis on original193x50 application/9496matrixPE program. Exact original row0 raw prefixes at declared widths/depths. Partial initialization; no matrix arithmetic, full weight loading or vocabulary inference acceptance.')
    (out / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(sdk_sha256=manifest['sdk_sha256'],
        files={p.name:digest(p) for p in out.iterdir() if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    import numpy as np
    from executor import verify
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform
    verify(root);os.chdir(root)
    config = json.loads((root / 'config.json').read_text())
    start = time.monotonic()
    def stage(name):
        row = dict(stage=name, elapsed_seconds=time.monotonic()-start)
        (root / 'stage.json').write_text(json.dumps(row)+'\n')
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(name, flush=True)
    command = ['cslc', 'layout.csl', '--arch=wse3', '--fabric-dims=200,52', '--fabric-offsets=4,1', '-o=out', '--memcpy', '--channels=1', '--dump-dsr-alloc-graph', '--max-parallelism=8']
    (root / 'compile-command.json').write_text(json.dumps(command)+'\n')
    stage('compile');subprocess.run(command, check=True)
    runner = SdkRuntime('out', get_platform(None, SimfabConfig(suppress_trace=True, num_threads=8, dump_core=True), SdkTarget.WSE3))
    ids = {name:runner.get_id(name) for name in ['weights', 'identity']}
    stage('load');runner.load();stage('run');runner.run()
    kwargs = dict(streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
    reports = []
    try:
        runner.launch('initialize', nonblock=False)
        identity = np.load(root / 'identity.npy').reshape(-1)
        assert identity.dtype == np.uint32 and len(identity) == 9650
        stage('h2d-identity-begin');runner.memcpy_h2d(ids['identity'], identity, 0, 0, 193, 50, 1, **kwargs);stage('h2d-identity-end')
        actual = np.zeros_like(identity)
        stage('d2h-identity-begin');runner.memcpy_d2h(actual, ids['identity'], 0, 0, 193, 50, 1, **kwargs);stage('d2h-identity-end')
        np.save(root / 'actual-identity.npy', actual);np.testing.assert_array_equal(actual, identity)
        assert (root / 'weights.u32.bin').stat().st_size == 9496*7168*4
        packed = np.memmap(root / 'weights.u32.bin', mode='c', dtype='<u4', shape=(9496, 7168))
        for index, case in enumerate(config['diagnostic_cases']):
            width, height, words = case['width'], case['height'], case['words']
            assert case.get('first_tile', 0) == 0 and case.get('source_word_offset', 0) == 0
            assert height == 1 and 1 <= width <= 192 and 1 <= words <= 7168
            expected = np.ascontiguousarray(packed[:width, :words]).reshape(-1)
            # Keep the source alive through exact roundtrip completion.
            elapsed = time.monotonic()
            stage(f'case-{index}-h2d-begin');runner.memcpy_h2d(ids['weights'], expected, 0, 0, width, height, words, **kwargs);stage(f'case-{index}-h2d-end')
            actual = np.zeros_like(expected)
            stage(f'case-{index}-d2h-begin');runner.memcpy_d2h(actual, ids['weights'], 0, 0, width, height, words, **kwargs);stage(f'case-{index}-d2h-end')
            roundtrip = time.monotonic()-elapsed
            path = root / f'actual-{index}.npy';np.save(path, actual)
            np.testing.assert_array_equal(actual, expected)
            reports.append(dict(case=case, exact=True, roundtrip_wall_seconds=roundtrip,
                source_bytes_sha256=hashlib.sha256(expected.tobytes()).hexdigest(), actual_sha256=digest(path)))
            (root / 'results.json').write_text(json.dumps(dict(success=False, diagnostic_only=True, cases=reports), indent=2)+'\n')
    finally:
        stage('stop');runner.stop()
    (root / 'results.json').write_text(json.dumps(dict(success=True, diagnostic_only=True, cases=reports, scope=config['scope']), indent=2)+'\n')
    stage('complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--from-fixture', type=Path)
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--short-first', action='store_true')
    args = parser.parse_args()
    if args.from_fixture:
        prepare(args.from_fixture.resolve(), args.project_root.resolve(), args.source_root.resolve(), args.short_first)
    elif args.worker:
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        execute(args.execute.resolve(), 1200)
