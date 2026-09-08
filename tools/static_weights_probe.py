"""Qualify ordinary CSL global weight initializers through the SDK loader.

This uses generated CSL source, never a debugger write or an ELF modification.
It is a separate initialization experiment, not a replacement for an active run.
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


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(source):
    import numpy as np
    sys.path.insert(0, str(ROOT / 'tools'))
    from bf16_transport import pack_bf16_pairs
    values = np.load(source / 'inputs.npz')
    packed = values['packed_weights'][0, 0].copy()
    words = pack_bf16_pairs(packed)
    matrix = (packed.astype(np.uint32) << 16).view(np.float32).T
    x = values['inputs'][0, 0, 0].copy()
    inputs = np.stack([x, -x, np.zeros_like(x), x])
    expected = inputs.astype(np.float64) @ matrix.astype(np.float64).T
    out = ROOT / 'evidence' / ('static-weights-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    np.savez(out / 'inputs.npz', weights=words, inputs=inputs, expected=expected)
    copies = [(Path(__file__), 'driver.py'), (ROOT / 'tools/sdk_probe.py', 'executor.py'),
              (ROOT / 'csl/programs/static_weights/kernel.csl', 'kernel.csl')]
    for path, name in copies:
        shutil.copy2(path, out / name)
    initializer = 'var weights=[7168]u32{' + ','.join(f'0x{int(word):08x}' for word in words) + '};\n'
    pe = '''param memcpy_params;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);
const clock=@import_module("<time>");
const kernel=@import_module("kernel.csl",.{.rows=128,.columns=112});
'''+initializer+'''
var input=@zeros([112]f32);var output=@zeros([128]f32);var progress=@zeros([1]u32);
var timing=@zeros([6]u16);var started=@zeros([3]u16);var ended=@zeros([3]u16);
fn compute() void {
 clock.enable_tsc();clock.get_timestamp(&started);
 kernel.compute(@ptrcast([*]u16,&weights),&input,&output);
 clock.get_timestamp(&ended);clock.disable_tsc();
 for(@range(u16,3))|i|{timing[i]=started[i];timing[i+3]=ended[i];}
 progress[0]+=1;sys.unblock_cmd_stream();
}
const wp:[*]u32=&weights;const ip:[*]f32=&input;const op:[*]f32=&output;const pp:[*]u32=&progress;const tp:[*]u16=&timing;
comptime{@export_symbol(wp,"weights");@export_symbol(ip,"input");@export_symbol(op,"output");@export_symbol(pp,"progress");@export_symbol(tp,"timing");@export_symbol(compute);}
'''
    (out / 'pe.csl').write_text(pe)
    exports = [('weights', 'u32'), ('input', 'f32'), ('output', 'f32'), ('progress', 'u32'), ('timing', 'u16')]
    layout = 'const memcpy=@import_module("<memcpy/get_params>",.{.width=1,.height=1});layout{@set_rectangle(1,1);@set_tile_code(0,0,"pe.csl",.{.memcpy_params=memcpy.get_params(0)});'
    layout += ''.join(f'@export_name("{name}",[*]{kind},false);' for name, kind in exports)
    (out / 'layout.csl').write_text(layout + '@export_name("compute",fn()void);}\n')
    config = dict(source_bundle=source.name, source_manifest_sha256=digest(source / 'manifest.json'),
                  source_inputs_sha256=digest(source / 'inputs.npz'),
                  cases=['canonical', 'negative', 'zero', 'repeat'], simulator_threads=1,
                  gate=dict(relative_l2=2e-6, relative_peak=3e-6),
                  scope='One actual128x112 BF16 tile through ordinary CSL global initializers and SDK load/run; full raw weight readback before/after four actual GEMVs. No debugger memory writes, no patched ELFs, no runtime weight H2D, not whole MLP/model or large-grid loading qualification.')
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    manifest = dict(sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',
                    files={p.name: digest(p) for p in out.iterdir() if p.is_file()})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(out, flush=True)


def worker(root):
    import numpy as np
    from executor import verify
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform
    verify(root)
    os.chdir(root)
    config = json.loads((root / 'config.json').read_text())
    data = np.load(root / 'inputs.npz')
    started = time.monotonic()

    def stage(name):
        value = dict(stage=name, elapsed_seconds=time.monotonic()-started)
        (root / 'stage.json').write_text(json.dumps(value) + '\n')
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value) + '\n')
        print(name, flush=True)

    command = ['cslc', 'layout.csl', '--arch=wse3', '--fabric-dims=8,3', '--fabric-offsets=4,1',
               '-o=out', '--memcpy', '--channels=1', '--dump-dsr-alloc-graph']
    (root / 'compile-command.json').write_text(json.dumps(command) + '\n')
    stage('compile')
    subprocess.run(command, check=True)
    runner = SdkRuntime('out', get_platform(None, SimfabConfig(suppress_trace=True, num_threads=config['simulator_threads'], dump_core=True), SdkTarget.WSE3))
    ids = {name: runner.get_id(name) for name in ['weights', 'input', 'output', 'progress', 'timing']}
    stage('load')
    runner.load()
    stage('run')
    runner.run()

    def read(name, count):
        value = np.zeros(count, np.float32 if name == 'output' else np.uint32)
        runner.memcpy_d2h(value, ids[name], 0, 0, 1, 1, count, streaming=False,
                          data_type=MemcpyDataType.MEMCPY_16BIT if name == 'timing' else MemcpyDataType.MEMCPY_32BIT,
                          order=MemcpyOrder.ROW_MAJOR, nonblock=False)
        return value

    reports = []
    try:
        stage('read-initial-weights')
        initial = read('weights', 7168)
        np.save(root / 'initial-weights.npy', initial)
        np.testing.assert_array_equal(initial, data['weights'])
        for epoch, name in enumerate(config['cases']):
            stage(name + '-input')
            runner.memcpy_h2d(ids['input'], data['inputs'][epoch].copy(), 0, 0, 1, 1, 112,
                              streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                              order=MemcpyOrder.ROW_MAJOR, nonblock=False)
            stage(name + '-compute-and-read')
            runner.launch('compute', nonblock=False)
            actual = {'output': read('output', 128), 'progress': read('progress', 1), 'timing': read('timing', 6)}
            file = root / f'actual-{epoch}.npz'
            np.savez(file, **actual)
            assert np.isfinite(actual['output']).all() and actual['progress'][0] == epoch + 1
            expected = data['expected'][epoch]
            error = actual['output'].astype(np.float64) - expected
            norm = float(np.linalg.norm(error) / max(np.linalg.norm(expected), 1e-30))
            peak = float(np.max(np.abs(error)) / max(np.max(np.abs(expected)), 1e-30))
            assert norm <= config['gate']['relative_l2'] and peak <= config['gate']['relative_peak']
            reports.append(dict(case=name, relative_l2=norm, relative_peak=peak, actual_sha256=digest(file)))
        stage('read-final-weights')
        final = read('weights', 7168)
        np.save(root / 'final-weights.npy', final)
        np.testing.assert_array_equal(final, initial)
    finally:
        runner.stop()
    result = dict(success=True, cases=reports, initial_weights_sha256=digest(root / 'initial-weights.npy'),
                  final_weights_sha256=digest(root / 'final-weights.npy'), scope=config['scope'])
    (root / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    stage('complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--execute', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare.resolve())
    elif args.worker:
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        execute(args.execute.resolve(), 1800)
