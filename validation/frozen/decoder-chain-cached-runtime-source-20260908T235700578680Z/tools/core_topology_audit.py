"""Offline full-topology core read cost and exact actual D2H comparison.

No SdkRuntime, compiler or simulator is constructed. The parent completed the
actual SDK execution; this audit reads its immutable stopped core only.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(parent, source, project):
    execution = json.loads((parent / 'execution.json').read_text())
    result = json.loads((parent / 'results.json').read_text())
    assert execution['success'] and result['success']
    assert execution['results_sha256'] == digest(parent / 'results.json')
    assert result['cases'][-1]['actual_sha256'] == digest(parent / 'actual-2-1.npz')
    resources = json.loads((parent / 'elf-memory.json').read_text())
    assert resources['passed']
    assert resources['manifest_sha256'] == execution['manifest_sha256']
    assert {p.name: digest(p) for p in (parent / 'out/bin').glob('*.elf')} == {
        row['elf']: row['sha256'] for row in resources['classes']}
    out = project / 'evidence' / ('core-topology-audit-' + datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    (out / 'bin').mkdir()
    for path in (parent / 'out/bin').glob('*.elf'):
        shutil.copy2(path, out / 'bin' / path.name)
    for path, name in [(Path(__file__), 'driver.py'), (source / 'tools/sdk_core.py', 'sdk_core.py'),
                       (source / 'tools/sdk_probe.py', 'executor.py'),
                       (parent / 'out.core', 'actual-runtime.core'),
                       (parent / 'execution.json', 'parent-execution.json'),
                       (parent / 'results.json', 'parent-results.json'),
                       (parent / 'elf-memory.json', 'parent-elf-memory.json'),
                       (parent / 'dsr-graphs.json', 'parent-dsr-graphs.json'),
                       (parent / 'actual-2-1.npz', 'actual-stream.npz'),
                       (parent / 'global-progress-2.npy', 'actual-progress.npy')]:
        shutil.copy2(path, out / name)
    (out / 'config.json').write_text(json.dumps(dict(parent=str(parent),
        parent_execution_sha256=digest(parent / 'execution.json'),
        parent_results_sha256=digest(parent / 'results.json'),
        core_sha256=digest(parent / 'out.core'), core_bytes=(parent / 'out.core').stat().st_size,
        output_symbol='$$csl_base_address$$0$$output',
        timeout_seconds=90, rss_limit_kib=4*1024*1024,
        timing_scope='SDK core construction then selected memory extraction/comparison/NPZ save; initial artifact hashing excluded. Host isolation is not enforced; concurrent work must be recorded separately. Not a full151936logit-read benchmark.',
        scope='Offline existing193x50 ET_CORE actual9650progress and49x128streamedwords comparison only. No new simulation, live snapshot, neural inference or hardware bandwidth.'), indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(sdk_sha256=execution['sdk_sha256'],
        files={str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    import numpy as np
    from executor import verify
    from sdk_core import StoppedCore
    verify(root)
    config = json.loads((root / 'config.json').read_text())
    for name, key in [('parent-execution.json', 'parent_execution_sha256'),
                      ('parent-results.json', 'parent_results_sha256'), ('actual-runtime.core', 'core_sha256')]:
        assert digest(root / name) == config[key]
    resources = json.loads((root / 'parent-elf-memory.json').read_text())
    dsr = json.loads((root / 'parent-dsr-graphs.json').read_text())
    assert dsr['passed'] and dsr['manifest_sha256'] == resources['manifest_sha256']
    assert dsr['elf_sha256'] == {row['elf']: row['sha256'] for row in resources['classes']}
    assert {p.name: digest(p) for p in (root / 'bin').glob('*.elf')} == {
        row['elf']: row['sha256'] for row in resources['classes']}
    start = time.monotonic()
    core = StoppedCore(root / 'actual-runtime.core', root / 'bin')
    load_seconds = time.monotonic()-start
    progress = np.array([int.from_bytes(core.read(x+4, y+1, 'progress', 4), 'little')
                         for y in range(50) for x in range(193)], dtype=np.uint32)
    np.testing.assert_array_equal(progress, np.load(root / 'actual-progress.npy'))
    assert np.all(progress == 1)  # Initial compiler memory was zero.
    output = np.stack([np.frombuffer(core.read(188, y+1, config['output_symbol'], 512), '<u4')
                       for y in range(49)])
    np.testing.assert_array_equal(output, np.load(root / 'actual-stream.npz')['output'])
    np.savez(root / 'actual-core.npz', progress=progress, output=output)
    (root / 'results.json').write_text(json.dumps(dict(success=True,
        core_sha256=config['core_sha256'], core_bytes=config['core_bytes'],
        load_seconds=load_seconds, load_and_read_seconds=time.monotonic()-start,
        actual_sha256=digest(root / 'actual-core.npz'), progress_words=progress.size,
        streamed_words=output.size, scope=config['scope'], timing_scope=config['timing_scope']), indent=2)+'\n')
    print('ACTUAL FULL TOPOLOGY CORE EXACT', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--from-stream', type=Path)
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.from_stream:
        prepare(args.from_stream.resolve(), args.source_root.resolve(), args.project_root.resolve())
    elif args.worker:
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        config = json.loads((args.execute / 'config.json').read_text())
        execute(args.execute.resolve(), config['timeout_seconds'], rss_limit_kib=config['rss_limit_kib'])
