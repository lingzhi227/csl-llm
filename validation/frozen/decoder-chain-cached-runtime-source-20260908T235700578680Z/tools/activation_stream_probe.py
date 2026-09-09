"""Frozen two-PE SDK layer-boundary transport probe; not model inference."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(source, project):
    import numpy as np
    out = project / 'evidence' / ('activation-stream-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    rng = np.random.default_rng(20260908)
    payloads = rng.integers(0, 1 << 32, (4, 896), dtype=np.uint32)
    # Include exact signed zeros, infinities, NaNs and finite float bit patterns.
    payloads[:, :8] = [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 1, 0xffffffff, 0x3f800000]
    np.savez(out / 'inputs.npz', payloads=payloads)
    files = [(Path(__file__), 'driver.py'), (source / 'tools/sdk_probe.py', 'executor.py'),
             (source / 'csl/programs/activation_stream/pe.csl', 'pe.csl'),
             (source / 'csl/runtime/activation_stream.csl', 'activation_stream.csl'),
             (source / 'csl/runtime/packets.csl', 'packets.csl')]
    for path, name in files:
        shutil.copy2(path, out / name)
    lines = ['const memcpy=@import_module("<memcpy/get_params>",.{.width=2,.height=1});layout{@set_rectangle(2,1);']
    for x in range(2):
        lines.append(f'@set_tile_code({x},0,"pe.csl",.{{.memcpy_params=memcpy.get_params({x}),.sender={"true" if x == 0 else "false"}}});')
    lines += [f'@export_name("{name}",[*]u32,false);' for name in ['data', 'control', 'status']]
    lines += [f'@export_name("{name}",fn()void);' for name in ['initialize', 'prepare', 'compute']]
    (out / 'layout.csl').write_text('\n'.join(lines + ['}']) + '\n')
    (out / 'config.json').write_text(json.dumps(dict(cases=[dict(epoch=e, command_first=c) for e, c in [(1, 0), (2, 1), (65535, 1), (1, 0)]],
        scope='Two PE 896-word packet framing, both command/data arrival orders, buffer lifetime, exact raw bits and reset after epoch65535. Not decoder arithmetic or full-model transport.'), indent=2) + '\n')
    (out / 'manifest.json').write_text(json.dumps(dict(sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',
        files={p.name: digest(p) for p in out.iterdir() if p.is_file()}), indent=2) + '\n')
    print(out, flush=True)


def compile_bundle(root):
    from executor import verify
    verify(root)
    os.chdir(root)
    command = ['cslc', 'layout.csl', '--arch=wse3', '--fabric-dims=9,3', '--fabric-offsets=4,1', '-o=out', '--memcpy', '--channels=1', '--max-parallelism=1', '--dump-dsr-alloc-graph']
    with (root / 'compile-command.json').open('x') as stream:
        json.dump(command, stream)
    subprocess.run(command, check=True)


def worker(root):
    import numpy as np
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform
    compile_bundle(root)
    config = json.loads((root / 'config.json').read_text())
    payloads = np.load(root / 'inputs.npz')['payloads']
    runner = SdkRuntime('out', get_platform(None, SimfabConfig(suppress_trace=True, num_threads=1, dump_core=True), SdkTarget.WSE3))
    ids = {name: runner.get_id(name) for name in ['data', 'control', 'status']}
    runner.load(); runner.run()
    kwargs = dict(streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
    reports = []
    try:
        runner.launch('initialize', nonblock=False)
        for index, case in enumerate(config['cases']):
            control = np.tile(np.array([case['epoch'], case['command_first']], np.uint32), 2)
            runner.memcpy_h2d(ids['control'], control, 0, 0, 2, 1, 2, **kwargs)
            runner.memcpy_h2d(ids['data'], payloads[index].copy(), 0, 0, 1, 1, 896, **kwargs)
            runner.launch('prepare', nonblock=False)
            runner.launch('compute', nonblock=False)
            # Source completion alone is not destination consumption. Inspect
            # destination's explicit completion before reading the payload.
            history = []
            for poll in range(64):
                status = np.zeros((2, 4), np.uint32)
                runner.memcpy_d2h(status.ravel(), ids['status'], 0, 0, 2, 1, 4, **kwargs)
                history.append(status.copy())
                if status[1, 0] == case['epoch']:
                    break
            received = np.zeros(896, np.uint32)
            runner.memcpy_d2h(received, ids['data'], 1, 0, 1, 1, 896, **kwargs)
            path = root / f'actual-{index}.npz'
            np.savez(path, received=received, status=status, completion_polls=np.array(history))
            np.testing.assert_array_equal(status, [[case['epoch'], 31, 896, 0], [case['epoch'], 31, 896, 1]])
            np.testing.assert_array_equal(received, payloads[index])
            reports.append(dict(case=case, exact=True, actual_sha256=digest(path), completion_polls=len(history)))
            (root / 'results.json').write_text(json.dumps(dict(success=False, cases=reports), indent=2) + '\n')
    finally:
        runner.stop()
    (root / 'results.json').write_text(json.dumps(dict(success=True, cases=reports, scope=config['scope']), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.source_root.resolve(), args.project_root.resolve())
    elif args.worker:
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        execute(args.execute.resolve(), 300, rss_limit_kib=2*1024*1024)
